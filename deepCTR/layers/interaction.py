#!/usr/bin/python
# -*- coding:utf-8 -*-
# @FileName  : interaction.py
# @Time      : 2022/3/10 14:13
# @Author    : weiguang
from operator import mod
import os
import sys
import itertools
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from activation import activation_layer
from core import Conv2dSame
from sequence import DynamicGRU, KMaxPooling

class FM(nn.Module):
    """
    因子分解机-二维交叉特征
    输入形状:
        - 3维矩阵 (batch_size, field_size, embedding_size)
    输出矩阵:
        - 2维矩阵 (batch_size, 1)
    References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """
    def __init__(self):
        super(FM, self).__init__()

    def forward(self, inputs):
        fm_input = inputs

        # 在dim=1矩阵上先求和并保留维度, 后求平方 (batch_size, 1, embedding_size)
        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)
        # fm_inputs求平方, 并在dim=1维度上求和 (batch_size, 1, embedding_size)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        # 在dim=2维度上求和, 并且不保留该维度 (batch_size, 1)
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)
        return cross_term

class BiInteractionPooling(nn.Module):
    """
     Bi-Interaction Layer used in Neural FM,compress the
     pairwise element-wise product of features into one single vector.
     神经FM中使用的双向交互层,将特征的成对元素乘积压缩为一个向量

     输入形状:
        - 3维矩阵,(batch_size, field_size, embedding_size)
     输出形状:
        - 3维矩阵, (batch_size, 1, embedding_size)
     References
        - [He X, Chua T S. Neural factorization machines for sparse predictive analytics[C]//Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval. ACM, 2017: 355-364.](http://arxiv.org/abs/1708.05027)
    """
    def __init__(self):
        super(BiInteractionPooling, self).__init__()

    def forward(self, inputs):
        concated_embeds_value = inputs
        # 在dim=1矩阵上先求和并保留维度, 后求平方 (batch_size, 1, embedding_size)
        square_of_sum = torch.pow(torch.sum(concated_embeds_value, dim=1, keepdim=True), 2)
        # concated_embeds_value求平方, 并在dim=1维度上求和 (batch_size, 1, embedding_size)
        sum_of_square = torch.sum(concated_embeds_value * concated_embeds_value, dim=1, keepdim=True)
        cross_term = 0.5 * (square_of_sum - sum_of_square)
        return cross_term

class SENETLayer(nn.Module):
    """
    应用于FiBiNET中的SENETLayer
    输出形状:
        - 3维数组, (batch_size, field_size, embedding_size)
    输入形状:
        - 3维数组, (batch_size, field_size, embedding_size)
    """
    def __init__(self, field_size, reduction_ratio=3, seed=1024, device='cpu'):
        """
        :param field_size: 特征组的数量
        :param reduction_ratio: 注意力机制输出维度
        :param seed: 种子
        :param device: 设备
        """
        super(SENETLayer, self).__init__()
        self.seed = seed
        self.field_size = field_size
        self.reduction_size = max(1, field_size // reduction_ratio)
        self.excitation = nn.Sequential(
            nn.Linear(self.field_size, self.reduction_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.reduction_size, self.field_size, bias=False),
            nn.ReLU()
        )
        self.to(device)

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))
        # inputs (batch_size,field_size,embedding_size) -> (batch_size, field_size)
        Z = torch.mean(inputs, dim=-1, out=None)
        # 经过两个线性层的变化(batch_size, field_size)
        A = self.excitation(Z)
        # inputs (batch_size,field_size,embedding_size)
        # torch.unsqueeze(A, dim=2) -> (batch_size,field_size,1)
        # 输出output形状:(batch_size,filed_size,embedding_size)
        V = torch.mul(inputs, torch.unsqueeze(A, dim=2))
        return V

class BilinearInteraction(nn.Module):
    """
    BilinearInteraction Layer used in FiBiNET.
    输入形状:
        - 3维数组,(batch_size,field_size, embedding_size)
    输出形状:
        - 3维数组,(batch_size,field_size*(field_size-1)/2, embedding_size)
    """

    def __init__(self, field_size, embedding_size, bilinear_type="interaction", seed=1024, device='cpu'):
        super(BilinearInteraction, self).__init__()
        self.bilinear_type = bilinear_type
        self.seed = seed
        self.bilinear = nn.ModuleList()

        if self.bilinear_type == "all":
            self.bilinear = nn.Linear(
                embedding_size, embedding_size, bias=False)
        elif self.bilinear_type == "each":
            for _ in range(field_size):
                self.bilinear.append(
                    nn.Linear(embedding_size, embedding_size, bias=False))
        elif self.bilinear_type == "interaction":
            for _, _ in itertools.combinations(range(field_size), 2):
                self.bilinear.append(
                    nn.Linear(embedding_size, embedding_size, bias=False))
        else:
            raise NotImplementedError
        self.to(device)

    def forward(self, inputs):
        # inputs (batch_size,field_size, embedding_size)
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))
        inputs = torch.split(inputs, 1, dim=1)
        if self.bilinear_type == "all":
            p = [torch.mul(self.bilinear(v_i), v_j)
                 for v_i, v_j in itertools.combinations(inputs, 2)]
        elif self.bilinear_type == "each":
            p = [torch.mul(self.bilinear[i](inputs[i]), inputs[j])
                 for i, j in itertools.combinations(range(len(inputs)), 2)]
        elif self.bilinear_type == "interaction":
            p = [torch.mul(bilinear(v[0]), v[1])
                 for v, bilinear in zip(itertools.combinations(inputs, 2), self.bilinear)]
        else:
            raise NotImplementedError
        # 输出形状为 (batch_size,field_size*(field_size-1)/2, embedding_size)
        return torch.cat(p, dim=1)

class CIN(nn.Module):
    """
    xDeepFM中使用的压缩交互网络
    输入形状:
        - 3维矩阵, (batch_size,field_size,embedding_size)`
    输出形状:
        - 2维矩阵, (batch_size, featuremap_num)
        - featuremap_num =  sum(self.layer_size[:-1]) // 2 + self.layer_size[-1]`` if ``split_half=True``,else  ``sum(layer_size)
    References
        - [Lian J, Zhou X, Zhang F, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems[J]. arXiv preprint arXiv:1803.05170, 2018.] (https://arxiv.org/pdf/1803.05170.pdf)
    """
    def __init__(self, field_size, layer_size=(128, 128), activation='relu', split_half=True, l2_reg=1e-5, seed=1024, device='cpu'):
        """
        :param field_size: 特征组的数量
        :param layer_size: list of int, 每一层的特征映射
        :param activation: 激活函数
        :param split_half: bool,如果设置为False,每个隐藏的特征映射的一半将连接到输出单元
        :param l2_reg:
        :param seed:
        :param device:
        :return:
        """
        super(CIN, self).__init__()
        if len(layer_size) == 0:
            raise ValueError(
                "layer_size must be a list(tuple) of length greater than 1")

        self.layer_size = layer_size
        self.field_nums = [field_size]
        self.split_half = split_half
        # 可配置不同的激活函数
        self.activation = activation_layer(activation)
        self.l2_reg = l2_reg
        self.seed = seed

        self.conv1ds = nn.ModuleList()
        # 遍历隐藏层layer_size
        for i, size in enumerate(self.layer_size):
            self.conv1ds.append(
                nn.Conv1d(self.field_nums[-1] * self.field_nums[0], size, 1)
            )

            if self.split_half:
                # 如果split_half=True,则layer_size必须为偶数
                if i != len(self.layer_size) - 1 and size % 2 > 0:
                    raise ValueError(
                        "layer_size must be even number except for the last layer when split_half=True")

                self.field_nums.append(size // 2)
            else:
                self.field_nums.append(size)

        #         for tensor in self.conv1ds:
        #             nn.init.normal_(tensor.weight, mean=0, std=init_std)
        self.to(device)

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))
        batch_size = inputs.shape[0]
        dim = inputs.shape[-1]
        hidden_nn_layers = [inputs]
        final_result = []

        for i, size in enumerate(self.layer_size):
            # x^(k-1) * x^0
            # 爱因斯坦求和约定 (b,h,d) (b,m,d) -> (b,h.m.d)
            x = torch.einsum(
                'bhd,bmd->bhmd', hidden_nn_layers[-1], hidden_nn_layers[0])
            # 将x-shape 转为 (batch_size , hi * m, dim)
            x = x.reshape(
                batch_size, hidden_nn_layers[-1].shape[1] * hidden_nn_layers[0].shape[1], dim)
            # x.shape = (batch_size , hi, dim)
            # 对x进行1维卷积
            x = self.conv1ds[i](x)

            if self.activation is None or self.activation == 'linear':
                curr_out = x
            else:
                curr_out = self.activation(x)

            # split_half为True则对curr_out执行切分
            # 否则赋值+初始化
            if self.split_half:
                if i != len(self.layer_size) - 1:
                    # split 按照[size//2, size//2]在dim=1维上进行切分
                    next_hidden, direct_connect = torch.split(
                        curr_out, 2 * [size // 2], 1)
                else:
                    direct_connect = curr_out
                    next_hidden = 0
            else:
                direct_connect = curr_out
                next_hidden = curr_out

            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)
        result = torch.cat(final_result, dim=1)
        # (batch_size, featuremap_num),dim=-1则合并完成
        result = torch.sum(result, -1)
        return result

class AFMLayer(nn.Module):
    """
    注意力因子分解机对没有线性项和偏差的成对（2 阶）特征交互进行建模
    输入形状:
        - 3维数组, (batch_size, 1, embedding_size)
    输出形状:
        - 2维数组, (bacth_size, 1)
    References
        - [Attentional Factorization Machines : Learning the Weight of Feature
        Interactions via Attention Networks](https://arxiv.org/pdf/1708.04617.pdf)
    """
    def __init__(self, in_features, attention_factor=4, l2_reg_w=0, dropout_rate=0, seed=1024, device='cpu'):
        """
        :param in_features: 输入特征维数
        :param attention_factor: attention网络输出维数
        :param l2_reg_w: 正则项
        :param dropout_rate:
        :param seed:
        :param device:
        """
        super(AFMLayer, self).__init__()
        self.attention_factor = attention_factor
        self.l2_reg_w = l2_reg_w
        self.dropout_rate = dropout_rate
        self.seed = seed
        embedding_size = in_features

        self.attention_W = nn.Parameter(torch.Tensor(
            embedding_size, self.attention_factor))
        self.attention_b = nn.Parameter(torch.Tensor(self.attention_factor))
        self.projection_h = nn.Parameter(
            torch.Tensor(self.attention_factor, 1))
        self.projection_p = nn.Parameter(torch.Tensor(embedding_size, 1))

        # 初始化为正态分布矩阵
        for tensor in [self.attention_W, self.projection_h, self.projection_p]:
            nn.init.xavier_normal_(tensor, )
        # 初始化为0矩阵
        for tensor in [self.attention_b]:
            nn.init.zeros_(tensor, )

        self.dropout = nn.Dropout(dropout_rate)
        self.to(device)

    def forward(self, inputs):
        embeds_vec_list = inputs
        row = []
        col = []
        # itertools.combinations实现排列组合,但是会自动过滤重复值
        for r,c in itertools.combinations(embeds_vec_list, 2):
            row.append(r)
            row.append(c)
        # 将行数据按照dim=1合并
        p = torch.cat(row, dim=1)
        # 将列数据按照dim=1合并
        q = torch.cat(col, dim=1)
        inner_product = p * q

        bi_interaction = inner_product
        # bi_interaction指定维度=-1与attention_W维度=0进行点乘
        # attention_W (embedding_size, attention_factor)
        attention_temp = F.relu(torch.tensordot(
            bi_interaction, self.attention_W, dims=([-1],[0])) + self.attention_b)

        self.normalized_att_score = F.softmax(torch.tensordot(
            attention_temp, self.projection_h, dims=([-1], [0])), dim=1)

        attention_output = torch.sum(
            self.normalized_att_score * bi_interaction, dim=1)

        attention_output = self.dropout(attention_output)  # training
        afm_out = torch.tensordot(
            attention_output, self.projection_p, dims=([-1], [0]))
        return afm_out

class InteractingLayer(nn.Module):
    """
    AutoInt Layer: 通过多头注意力机制的不同领域的相关性模型
    输入形状:
        - 3维数组,形状为(batch_size, field_size, embedding_size)
    输出形状:
        - 3维数组,形状为(batch_size, field_size, embedding_size)
    References
        - [Song W, Shi C, Xiao Z, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks[J]. arXiv preprint arXiv:1810.11921, 2018.](https://arxiv.org/abs/1810.11921)
    """
    def __init__(self, embedding_size, head_num=2, use_res=True, scaling=False, seed=1024, device='cpu'):
        """
        :param embedding_size: 特征维度
        :param head_num: 多头注意力机制数量
        :param use_res:
        :param scaling:
        :param seed:
        :param device:
        """
        super(InteractingLayer, self).__init__()
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        if embedding_size % head_num != 0:
            raise ValueError('embedding_size is not a integer multiple of head_num!')
        self.att_embedding_size = embedding_size // head_num
        self.head_num = head_num
        self.use_res = use_res
        self.scaling = scaling
        self.seed = seed

        # 构建query、key与value矩阵
        self.W_query = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_key = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_value = nn.Parameter(torch.Tensor(embedding_size, embedding_size))

        if self.use_res:
            self.W_Res = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.05)
        self.to(device)

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))

        # inputs作为输入,分别与权重矩阵W_query、W_key、W_value点乘
        # 指定inputs为dim=-1, 权重dim=0进行点乘 如 (2,1,3) dot (3,5,2) - > (2,1,5,2)
        # (None, F, D)
        querys = torch.tensordot(inputs, self.W_query, dims=([-1], [0]))
        keys = torch.tensordot(inputs, self.W_key, dims=([-1], [0]))
        values = torch.tensordot(inputs, self.W_value, dims=([-1],[0]))

        # 按照multi_head中head_num进行切割
        # 先按照dim=2进行均分,然后进行合并,例如torch.split(6,4,4, dim=2)->2个(6,4,2)进行stack->(2,6,4,2)
        # (head_num, None, F, D//head_num)
        querys = torch.stack(torch.split(querys, self.att_embedding_size, dim=2))
        keys = torch.stack(torch.split(keys, self.att_embedding_size, dim=2))
        values = torch.stack(torch.split(values, self.att_embedding_size, dim=2))

        # (head_num, None, F, F)
        inner_product = torch.einsum('bnik,bnjk->bnij', querys, keys)
        if self.scaling:
            inner_product /= self.att_embedding_size ** 0.5
        # (head_num, None, F, F)
        self.normalized_att_scores = F.softmax(inner_product, dim=-1)
        # (head_num, None, F, D/head_num)
        result = torch.matmul(self.normalized_att_scores, values)

        result = torch.cat(torch.split(result, 1, ), dim=-1)
        # (None, F, D)
        result = torch.squeeze(result, dim=0)
        if self.use_res:
            result += torch.tensordot(inputs, self.W_Res, dims=([-1], [0]))
        return result

class CrossNet(nn.Module):
    """
    Deep&Cross Network模型的Cross Network部分 它倾向于低度和高度交叉特征
    输入形状:
        - 2维数组, (batch_size, units)
    输出形状:
        - 2维数组, (batch_size, units)

    References
        - [Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12.](https://arxiv.org/abs/1708.05123)
        - [Wang R, Shivanna R, Cheng D Z, et al. DCN-M: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems[J]. 2020.](https://arxiv.org/abs/2008.13535)
    """
    def __init__(self, in_features, layer_num=2, parameterization='vector', seed=1024, device='cpu'):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        self.parameterization = parameterization
        if self.parameterization == 'vector':
            # weight in DCN (in_features, 1)
            self.kernels = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))
        elif self.parameterization == 'maxtrix':
            # weight matrix in DCN-M.  (in_features, in_features)
            self.kernels = nn.Parameter(torch.Tensor(self.layer_num, in_features, in_features))
        else:
            raise ValueError("parameterization should be 'vector' or 'matrix'")

        self.bias = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))

        for i in range(self.kernels.shape[0]):
            nn.init.xavier_normal_(self.kernels[i])
        for i in range(self.bias.shape[0]):
            nn.init.zeros_(self.bias[i])

        self.to(device)


    def forward(self, inputs):
        # inputs -> (batch_size, units)
        # x_0 = (batch_size, units, 1)
        x_0 = inputs.unsqueeze(2)
        x_l = x_0
        for i in range(self.layer_num):
            if self.parameterization == 'vector':
                xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1],[0]))
                dot_ = torch.matmul(x_0, xl_w)
                x_l = dot_ + self.bias[i] + x_l
            elif self.parameterization == 'matrix':
                # W * xi (in_features, 1)
                xl_w = torch.matmul(self.kernels[i], x_l)
                # W * xi + b
                dot_ = xl_w + self.bias[i]
                # x0 · (W * xi + b) +xl  Hadamard-product
                x_l = x_0 * dot_ + x_l
            else:
                raise ValueError("parameterization should be 'vector' or 'matrix'")
        x_l = torch.squeeze(x_l, dim=2)
        return x_l

class CrossNetMix(nn.Module):
    """
    Cross Network中的DCN-Mix模型, 对于DCN-M有以下提升:
    1 添加了MOE用于学习在不同子空间中特征的相互影响
    2 在低维空间中添加了非线性的转换

    输入形状:
        - 2维数组, (batch_size, units)
    输出形状:
        - 2维数组, (batch_size, units)
    
    References
        - [Wang R, Shivanna R, Cheng D Z, et al. DCN-M: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems[J]. 2020.](https://arxiv.org/abs/2008.13535)
    """
    def __init__(self, in_features, low_rank=32, num_experts=4, layer_num=2, device='cpu'):
        """
        Args:
            in_features (_type_): 输入特征的维度
            low_rank (int, optional): 低维空间中的维度. Defaults to 32.
            num_experts (int, optional): exports数量. Defaults to 4.
            layer_num (int, optional): cross layer数量. Defaults to 2.
            device (str, optional): 运行的设备. Defaults to 'cpu'.
        """
        super(CrossNetMix, self).__init__()
        self.layer_num = layer_num
        self.num_experts = num_experts
        
        # U: (in_features, low_rank)
        self.U_list = nn.Parameter(torch.Tensor(self.layer_num, num_experts, in_features, low_rank))
        # V: (in_features, low_rank)
        self.V_list = nn.Parameter(torch.Tensor(self.layer_num, num_experts, in_features, low_rank))
        # C: (low_rank, low_rank)
        self.C_list = nn.Parameter(torch.Tensor(self.layer_num, num_experts, low_rank, low_rank))
        self.gating = nn.ModuleList([nn.Linear(in_features, 1, bias=False) for i in range(self.num_experts)])
        self.bias = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))

        init_para_list = [self.U_list, self.V_list, self.C_list]
        for para in init_para_list:
            for i in range(self.layer_num):
                nn.init.xavier_normal_(para[i])

        for i in range(len(self.bias)):
            nn.init.zeros_(self.bias[i])
        
        self.to(device)
    
    def forward(self, inputs):
        # 维度为(bs, in_features, 1)
        x_0 = inputs.unsqueeze(2)
        x_l = x_0
        for i in range(self.layer_num):
            output_of_experts = []
            gating_score_of_experts = []
            for expert_id in range(self.num_experts):
                # (1) G(x_l)
                # compute the gating score by x_l
                gating_score_of_experts.append(self.gating[expert_id](x_l.squeeze(2)))

                # (2) E(x_l)
                # project the input x_l to $\mathbb{R}^{r}$
                v_x = torch.matmul(self.V_list[i][expert_id].t(), x_l)  # (bs, low_rank, 1)

                # nonlinear activation in low rank space
                v_x = torch.tanh(v_x)
                v_x = torch.matmul(self.C_list[i][expert_id], v_x)
                v_x = torch.tanh(v_x)

                # project back to $\mathbb{R}^{d}$
                uv_x = torch.matmul(self.U_list[i][expert_id], v_x)  # (bs, in_features, 1)

                dot_ = uv_x + self.bias[i]
                dot_ = x_0 * dot_  # Hadamard-product

                output_of_experts.append(dot_.squeeze(2))
            # (3) mixture of low-rank experts
            output_of_experts = torch.stack(output_of_experts, 2)  # (bs, in_features, num_experts)
            gating_score_of_experts = torch.stack(gating_score_of_experts, 1)  # (bs, num_experts, 1)
            moe_out = torch.matmul(output_of_experts, gating_score_of_experts.softmax(1))
            x_l = moe_out + x_l  # (bs, in_features, 1)

        x_l = x_l.squeeze()  # (bs, in_features)
        return x_l

class InnerProductLayer(nn.Module):
    """
    PNN 中使用的 InnerProduct 层, 用于计算特征向量之间的元素乘积或内积
    输入形状:
        - 3维数组列表, [(batch_size, 1, embedding_size),...]
    输出形状:
        - reduce_sum=True, 3维数组, (batch_size, N*(N-1)/2, 1)
        - reduce_sum=False, 3维数组, (batch_size, N*(N-1)/2, embedding_size)
    
    References
        - [Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response prediction[C]//
        Data Mining (ICDM), 2016 IEEE 16th International Conference on. IEEE, 2016: 1149-1154.]
        (https://arxiv.org/pdf/1611.00144.pdf)
    """
    def __init__(self, reduce_sum=True, device='cpu'):
        super(InnerProductLayer, self).__init__()
        self.reduce_sum = reduce_sum
        self.to(device)
    
    def forward(self, inputs):
        embed_list = inputs
        row = []
        col = []
        num_inputs = len(embed_list)

        for i in range(num_inputs - 1):
            for j in range(i + 1, num_inputs):
                row.append(i)
                col.append(j)
        # size -> (batch, num_pairs, k)
        p = torch.cat([embed_list[idx] for idx in row], dim=1)
        q = torch.cat([embed_list[idx] for idx in col], dim=1)

        inner_product = p * q
        if self.reduce_sum:
            inner_product = torch.sum(inner_product, dim=2, keepdim=True)
        return inner_product


class OutterProductLayer(nn.Module):
    """
    PNN中的OutterProduct Layer
    输入形状:
        - 3维数组列表, [(batch_size, 1, embedding_size),...]
    输出形状:
        - 2维数组, (batch_size, N*(N-1))
    References
        - [Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response prediction[C]//Data Mining (ICDM), 2016 IEEE 16th International Conference on. IEEE, 2016: 1149-1154.](https://arxiv.org/pdf/1611.00144.pdf)
    """
    def __init__(self, field_size, embedding_size, kernel_type='mat', seed=1024, device='cpu'):
        super(OutterProductLayer, self).__init__()
        self.kernel_type = kernel_type

        num_inputs = field_size
        num_pairs = int(num_inputs * (num_inputs - 1) / 2)
        embed_size = embedding_size
        
        if self.kernel_type == "mat":
            self.kernel = nn.Parameter(torch.Tensor(embed_size, num_pairs, embed_size))
        elif self.kernel_type == "vec":
            self.kernel = nn.Parameter(torch.Tensor(num_pairs, embed_size))
        elif self.kernel_type == "num":
            self.kernel = nn.Parameter(torch.Tensor(num_pairs, 1))
        
        nn.init.uniform_(self.kernel)
        self.to(device)
    
    def forward(self, inputs):
        embed_list = inputs
        row = []
        col = []
        num_inputs = len(embed_list)
        for i in range(num_inputs - 1):
            for j in range(i + 1, num_inputs):
                row.append(i)
                col.append(j)
        p = torch.cat([embed_list[idx]
                       for idx in row], dim=1)  # batch num_pairs k
        q = torch.cat([embed_list[idx] for idx in col], dim=1)
        
        if self.kernel_type == "mat":
            p.unsqueeze_(dim=1)
            # k     k* pair* k
            # batch * pair
            kp = torch.sum(
                # batch * pair * k
                torch.mul(
                    # batch * pair * k
                    torch.transpose(
                        # batch * k * pair
                        torch.sum(
                            # batch * k * pair * k
                            torch.mul(p, self.kernel),
                        dim=-1),
                    2, 1),
                q),
            dim=-1)
        else:
            # 1 * pair * (k or 1)
            k = torch.unsqueeze(self.kernel, 0)
            # batch * pair
            kp = torch.sum(p * q * k, dim=-1)
            # p q # b * p * k
        return kp

class ConvLayer(nn.Module):
    """
    CCPM中卷积核
    输入形状:
        - 3维数组列表, (batch_size, 1, field_size, embedding_size)
    输出形状:
        - 3维数组列表, (batch_size, last_filters, pooling_size, embedding_size)
    Reference:
        - Liu Q, Yu F, Wu S, et al. A convolutional click prediction model[C]//Proceedings of the 24th ACM International on Conference on Information and Knowledge Management. ACM, 2015: 1743-1746.(http://ir.ia.ac.cn/bitstream/173211/12337/1/A%20Convolutional%20Click%20Prediction%20Model.pdf)
    """
    def __init__(self, field_size, conv_kernel_width, conv_filters, device='cpu'):
        """
        Args:
            field_size (_type_): 特征组的个数
            conv_kernel_width (_type_): 每个kernel对应的宽度
            conv_filters (_type_): 每一个conv layer对应的过滤器的个数
            device (str, optional): 设备
        """
        super(ConvLayer, self).__init__()
        self.device = device
        module_list = []
        n = int(field_size)
        l = len(conv_filters)
        field_shape = n
        for i in range(1, l+1):
            if i == 1:
                in_channels = 1
            else:
                in_channels = conv_filters[i - 2]
            out_channels = conv_filters[i - 1]
            width = conv_kernel_width[i - 1]
            k = max(1, int(1 - pow(i / l, l - i) * n)) if i < l else 3
            # Conv2dSame提供了一种same方式的Conv2d
            module_list.append(Conv2dSame(in_channels=in_channels, out_channels=out_channels, \
                kernel_size=(width, 1), stride=1).to(self.device))
            module_list.append(torch.nn.Tanh().to(self.device))

            # 取Top-k的Pooling方式
            module_list.append(KMaxPooling(k=min(k, field_shape), axis=2, device=self.device).to(self.device))
            field_shape = min(k ,field_shape)
        # 采用Sequential搭建网络结构
        # 第一步: Conv2dSame->same卷积方式
        # 第二步: Tanh激活函数
        # 第三步: KMaxPooling池化
        self.conv_layer = nn.Sequential(*module_list)
        self.to(device)
        self.field_shape = field_shape

    def forward(self ,inputs):
        return self.conv_layer(inputs)

class LogTransformLayer(nn.Module):
    """
    自适应分解网络中的对数变换层，任意阶交叉特征模型
    输入形状:
        - 3维数组, (batch_size, field_size, embedding_size)
    输出形状:
        - 2维数组, (batch, ltl_hidden_size*embedding_size)
    """
    def __init__(self, field_size, embedding_size, ltl_hidden_size):
        """
        Args:
            field_size (_type_): 特征数组个数
            embedding_size (_type_): 特征维度
            ltl_hidden_size (_type_): AFN中对数神经元个数
        """
        super(LogTransformLayer, self).__init__()
        # 添加weights与biases
        self.ltl_weights = nn.Parameter(torch.Tensor(field_size, ltl_hidden_size))
        self.ltl_biases = nn.Parameter(torch.Tensor(1, 1, ltl_hidden_size))
        # 添加两个BN层
        self.bn = nn.ModuleList([nn.BatchNorm1d(embedding_size) for i in range(2)])
        nn.init.normal_(self.ltl_weights, mean=0.0, std=0.1)
        nn.init.zeros_(self.ltl_biases, )

    def forward(self, inputs):
        # 避免数值溢出
        afn_input = torch.clamp(torch.abs(inputs), min=1e-7, max=float("Inf"))
        # 对数组转置
        afn_input_trans = torch.transpose(afn_input, 1, 2)
        # 对数变化的layer
        ltl_result = torch.log(afn_input_trans)
        ltl_result = self.bn[0](ltl_result)
        ltl_result = torch.matmul(ltl_result, self.ltl_weights) + self.ltl_biases
        ltl_result = torch.exp(ltl_result)
        ltl_result = self.bn[1](ltl_result)
        ltl_result = torch.flatten(ltl_result, start_dim=1)
        return ltl_result

if __name__ == '__main__':
    batch_size, field_size, embedding_size = 128, 20, 32
    inputs = torch.randn((batch_size, field_size, embedding_size))
    # FM
    fm = FM()
    print('fm:',fm(inputs).shape)  # (batch_size, 1)

    # BiInteractionPooling
    BIPooling = BiInteractionPooling()
    print('BIPooling:',BIPooling(inputs).shape)  # (batch_size, 1, embedding_size)

    # SENETLayer
    senelt_layer = SENETLayer(field_size, reduction_ratio=3, seed=1024, device='cpu')
    print('senelt_layer:',senelt_layer(inputs).shape)   # (batch_size, field_size, embedding_size)

    # BilinearInteraction
    bilinear_interaction = BilinearInteraction(field_size, embedding_size, bilinear_type="interaction", seed=1024, device='cpu')
    print('bilinear_interaction:', bilinear_interaction(inputs).shape)  # (batch_size, field_size*(field_size)/2, embedding_size)

    # CIN
    cin = CIN(field_size, layer_size=(128, 128), activation='relu', split_half=True, l2_reg=1e-5, seed=1024, device='cpu')
    print('cin:', cin(inputs).shape)  # (batch_size, sum(layer_size[:-1]//2 + self.layer_size[-1]))  128//2 + 128 = 192

    # # AFM layer
    # afm_layer = AFMLayer(32, attention_factor=4, l2_reg_w=0)
    # afm_inputs = torch.randn((batch_size, 1, embedding_size))
    # print('afm layer:', AFMLayer(afm_inputs).shape)

    # Interaction layer
    interaction_layer = InteractingLayer(embedding_size, head_num=2, use_res=True, scaling=False, seed=1024, device='cpu')
    print('interaction layer:', interaction_layer(inputs).shape)

    # crossNet
    # in_features与Unit需要一致
    cross_net = CrossNet(in_features=8, layer_num=2, parameterization='vector', seed=1024, device='cpu')
    cross_inputs = torch.randn((128, 8))
    print('cross net:', cross_net(cross_inputs).shape)

    # crossMixNet
    cross_mix_net = CrossNetMix(in_features=8, low_rank=32, num_experts=4, layer_num=2, device='cpu')
    cross_mix_inputs = torch.randn((128, 8))
    print('cross mix net:', cross_mix_net(cross_mix_inputs).shape)

    # innerProductLayer
    inner_product_layer = InnerProductLayer(reduce_sum=False, device='cpu')
    inner_inputs = [torch.randn((batch_size, 1, embedding_size)) for _i in range(8)]
    print('inner product layer:', inner_product_layer(inner_inputs).shape)
    
    # outter_product_layer
    outter_product_layer = OutterProductLayer(field_size=8, embedding_size=32, kernel_type="mat")
    outter_inputs = [torch.randn((batch_size, 1, 32)) for _ in range(8)]
    print("outter product layer:", outter_product_layer(outter_inputs).shape)

    # ConvLayer
    conv_layer = ConvLayer(field_size=8, conv_kernel_width=[2,3,4], conv_filters=[2,3,4])
    # (batch_size, 1, field_size, embedding_size)
    conv_inputs = torch.randn((128, 1, 8, 32))
    print('conv layer:', conv_layer(conv_inputs).shape)

    # LogTransformLayer
    log_transform_layer = LogTransformLayer(field_size=8, embedding_size=32, ltl_hidden_size=16)
    # (batch_size, field_size, embedding_size)
    log_trans_inputs = torch.randn((128, 8, 32))
    print("log transform layer:", log_transform_layer(log_trans_inputs).shape)



