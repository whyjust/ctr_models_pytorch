#!/usr/bin/python
# -*- coding:utf-8 -*-
# @FileName  : sequence.py
# @Time      : 2022/3/7 21:11
# @Author    : weiguang
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core import LocalActivationUnit

class SequencePoolingLayer(nn.Module):
    def __init__(self, mode='mean', supports_masking=False, device='cpu'):
        """
        SequencePoolingLayer用于对变长序列或多值序列的pooling计算(sum/mean/max)
        :param mode: 支持的pooling模式
        :param supports_masking: 是否支持mask
        :param device: 设备类型

        输入形状：一组tensor [seq_value, seq_len]
         - seq_value: 3维数组 [batch_size, T, embedding_size]
         - seq_len  : 2维数组 [batch_size, 1], 表示每个序列的有效长度
        输出形状：
         - 经过pooling之后的3维数组，[batch_size, 1, embedding_size]
        """
        super(SequencePoolingLayer, self).__init__()
        if mode not in ['sum','mean','max']:
            raise ValueError('parameter mode should in [sum,max,mean]')
        self.supports_masking = supports_masking
        self.mode = mode
        self.device = device
        self.eps = torch.FloatTensor([1e-8]).to(device)
        self.to(device)

    def _sequence_mask(self, lengths, maxlen=None, dtype=torch.bool):
        if maxlen is None:
            maxlen = lengths.max()
        # lengths.device获取lengths对应的device属性
        # row_vector: size->(maxlen,1), 例如(12,1)
        row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
        # 将lengths扩增一个维度，例如(2,3)->(2,3,1)
        matrix = torch.unsqueeze(lengths, dim=-1)
        # 将row_vector与maxtrix比较
        # 维度扩增为(2,3,maxlen), row_vector值小于matrix对应维度值，则返回True否则返回False
        mask = row_vector < matrix
        mask.type(dtype)
        return mask

    def forward(self, seq_value_len_list):
        if self.supports_masking:
            uiseq_embed_list, mask = seq_value_len_list  # [B, T, E], [B, 1]
            mask = mask.float()
            user_behavior_length = torch.sum(mask, dim=-1, keepdim=True)
            mask = mask.unsqueeze(2)
        else:
            uiseq_embed_list, user_behavior_length = seq_value_len_list  # [B, T, E], [B, 1]
            # 得到根据np.arange(0,maxlen,1)与user_behavior_length比较后的mask矩阵
            # 维度在user_behavior_length: (B, 1) -> (B, 1, maxlen)
            mask = self._sequence_mask(user_behavior_length, maxlen=uiseq_embed_list.shape[1],
                                       dtype=torch.float32)  # [B, 1, maxlen]
            mask = torch.transpose(mask, 1, 2)  # [B, maxlen, 1]

        embedding_size = uiseq_embed_list.shape[-1]
        # 在dim=2的维度上复制embedding_size份数据，维度由 (B, maxlen, 1) -> [B, maxlen, E]
        mask = torch.repeat_interleave(mask, embedding_size, dim=2)  # [B, maxlen, E]

        if self.mode == 'mask':
            hist = uiseq_embed_list - (1 - mask) * 1e9
            hist = torch.max(hist, dim=1, keepdim=True)[0]
            return hist
        hist = uiseq_embed_list * mask.float()  # [B, maxlen, E]
        hist = torch.sum(hist, dim=1, keepdim=False)    # [B, E]

        if self.mode == 'mean':
            self.eps = self.eps.to(user_behavior_length.device)
            # user_behavior_length.type(torch.float32)将user_behavior_length值转为float32
            # hist:(B, E) user_behavior_length:(B,1) -> (B, E)
            hist = torch.div(hist, user_behavior_length.type(torch.float32) + self.eps)

        # 在dim=1的维度上新增一维
        hist = torch.unsqueeze(hist, dim=1)     # (B, 1, E)
        return hist

class AttentionSequencePoolingLayer(nn.Module):
    """
    输入形状：3个tensor组成的数组，[query, keys, keys_length]
        - query: 3维数组， (batch_size, 1, embedding_size)
        - keys: 3维数组， (batch_size, T, embedding_size)
        - keys_length: 2维数组， (batch_size, 1)
    输出形状：
        - 3维数组, (batch_size, 1, embedding_size)
    """
    def __init__(self, att_hidden_units=(80, 40), att_activation='sigmoid', weight_normalization=False,
                 return_score=False, supports_masking=False, embedding_dim=4, **kwargs):
        """
        应用于DIN与DIEN的注意力池化机制
        :param att_hidden_units: 注意力机制层数与对应的神经元数
        :param att_activation: 注意力机制激活函数
        :param weight_normalization: 权重标准化
        :param return_score:
        :param supports_masking: 如果为Ture，输入需要支持mask
        :param embedding_dim:
        :param kwargs:
        """
        super(AttentionSequencePoolingLayer, self).__init__()
        self.return_score = return_score
        self.weight_normalization = weight_normalization
        self.supports_masking = supports_masking
        # 根据用户行为与query经过DNN层得到的attention_score
        self.local_att = LocalActivationUnit(hidden_units=att_hidden_units, embedding_dim=embedding_dim,
                                             activation=att_activation,
                                             dropout_rate=0, use_bn=False)

    def forward(self, query, keys, keys_length, mask=None):
        """
        :param query: 3维数组， (batch_size, 1, embedding_size)
        :param keys: 3维数组， (batch_size, T, embedding_size)
        :param keys_length: 2维数组， (batch_size, 1)
        :param mask:
        :return:
        """
        batch_size, max_length, _ = keys.size()

        # mask
        if self.supports_masking:
            if mask is None:
                raise ValueError('when support_masking=True, input must support masking!')
            keys_masks = mask.unsqueeze(1)  # (B, 1, T)
        else:
            keys_masks = torch.arange(max_length, device=keys_length.device, dtype=keys_length.dtype)\
                              .repeat(batch_size, 1)     # (B, T)
            # view将keys_length: (B,1) -> (B,1)
            keys_masks = keys_masks < keys_length.view(-1, 1)     # (B，T)
            # 将得到的bool矩阵新增一维
            keys_masks = keys_masks.unsqueeze(1)  # (B, 1, T)

        attention_score = self.local_att(query, keys)   # (B, T, 1)
        outputs = torch.transpose(attention_score, 1, 2)    # (B, 1, T)

        if self.weight_normalization:
            padding = torch.ones_like(outputs) * (-2**32 + 1)
        else:
            padding = torch.zeros_like(outputs)

        # 对keys_mask小于keys_length中元素也就是为True，取outputs值，否则取对应padding的值
        outputs = torch.where(keys_masks, outputs, padding)  # (B, 1, T)

        # scale
        # outputs = outputs / (keys.shape[-1] ** 0.5)

        if self.weight_normalization:
            # 按照dim=-1最后一维做softmax转换->(B, 1, T)
            outputs = F.softmax(outputs, dim=-1)

        if not self.return_score:
            # outputs=(B,1,T), keys=(B,T,E) -> (B, 1, E)
            # weight sum
            outputs = torch.matmul(outputs, keys)
        return outputs


class KMaxPooling(nn.Module):
    """
    在指定维度上选择TopK个值
    输入形状：
        - n维数组，(batch_size, ..., input_dim)
    输出形状：
        - n维数组，(batch_size, ..., output_dim)
    """
    def __init__(self, k, axis, device='cpu'):
        """
        :param k:  按照指定的维度选择K个值
        :param axis: 指定的维度
        :param device: 设备
        """
        super(KMaxPooling, self).__init__()
        self.k = k
        self.axis = axis
        self.to(device)

    def forward(self, inputs):
        if self.axis < 0 or self.axis >= len(inputs.shape):
            raise ValueError("axis must be 0~%d,now is %d" %
                             (len(inputs.shape) - 1, self.axis))

        if self.k < 1 or self.k > inputs.shape[self.axis]:
            raise ValueError("k must be in 1 ~ %d,now k is %d" %
                             (inputs.shape[self.axis], self.k))

        out = torch.topk(inputs, k=self.k, dim=self.axis, sorted=True)[0]
        return out

class AGRUCell(nn.Module):
    """
    基于GRU的Attention

    Reference:
        -  Deep Interest Evolution Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1809.03672, 2018.
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(AGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # (W_ir|W_iz|W_ih),3组权重值的注册
        # weight注册变量
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.register_parameter('weight_ih', self.weight_ih)
        # (W_hr|W_hz|W_hh)，3组权重值的注册
        # weight注册变量
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.register_parameter('weight_hh', self.weight_hh)
        if bias:
            # (b_ir|b_iz|b_ih)，3组偏置值的注册
            self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_ih)
            # (b_hr|b_hz|b_hh) 3组偏置值的注册
            self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_hh', self.bias_hh)
            for tensor in [self.bias_ih, self.bias_hh]:
                # 偏置值初始化为0
                nn.init.zeros_(tensor)
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

    def forward(self, inputs, hx, att_score):
        # 做一个线性转换 gi = Input * W + b
        gi = F.linear(inputs, self.weight_ih, self.bias_hh)
        gh = F.linear(hx, self.weight_hh, self.bias_hh)
        # 将映射后的值拆分成3个等长的矩阵
        i_r, _, i_n = gi.chunk(3, 1)
        h_r, _, h_n = gh.chunk(3, 1)

        # 重置门
        reset_gate = torch.sigmoid(i_r + h_r)
        # update_gate = torch.sigmoid(i_z + h_z)
        # 更新状态
        new_state = torch.tanh(i_n + reset_gate * h_n)

        att_score = att_score.view(-1,1)
        hy = (1. - att_score) * hx + att_score * new_state
        return hy

class AUGRUCell(nn.Module):
    """
    基于更新门的GRU的Attention

    Reference:
        -  Deep Interest Evolution Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1809.03672, 2018.
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(AUGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # (W_ir|W_iz|W_ih),3组权重值的注册
        # weight注册变量
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.register_parameter('weight_ih', self.weight_ih)
        # (W_hr|W_hz|W_hh)，3组权重值的注册
        # weight注册变量
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.register_parameter('weight_hh', self.weight_hh)
        if bias:
            # (b_ir|b_iz|b_ih)，3组偏置值的注册
            self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_ih)
            # (b_hr|b_hz|b_hh) 3组偏置值的注册
            self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_hh)
            for tensor in [self.bias_ih, self.bias_hh]:
                nn.init.zeros_(tensor, )
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

    def forward(self, inputs, hx, att_score):
        # 做一个线性转换 gi = Input * W + b
        gi = F.linear(inputs, self.weight_ih, self.bias_ih)
        gh = F.linear(hx, self.weight_hh, self.bias_hh)
        # 将映射后的值拆分成3个等长的矩阵
        i_r, i_z, i_n = gi.chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)

        # reset_gate
        reset_gate = torch.sigmoid(i_r + h_r)
        # 更新门
        update_date = torch.sigmoid(i_z + h_z)
        # 新状态
        new_state = torch.tanh(i_n + reset_gate * h_n)

        att_score = att_score.view(-1, 1)
        update_date = att_score * update_date
        hy = (1. - update_date) * hx + update_date * new_state
        return hy

class DynamicGRU(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, gru_type='AGRU'):
        super(DynamicGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        if gru_type == 'AGRU':
            self.rnn = AGRUCell(input_size, hidden_size, bias)
        elif gru_type == 'AUGRU':
            self.rnn = AUGRUCell(input_size, hidden_size, bias)

    def forward(self, inputs, att_scores=None, hx=None):
        if not isinstance(inputs, PackedSequence) or not isinstance(att_scores, PackedSequence):
            raise NotImplementedError("DynamicGRU only supports packed input and att_scores")

        inputs, batch_sizes, sorted_indices, unsorted_indices = inputs
        att_scores, _, _, _ = att_scores

        max_batch_size = int(batch_sizes[0])
        # 如果hx=None,则初始化
        if hx is None:
            hx = torch.zeros(max_batch_size, self.hidden_size,
                             dtype=inputs.dtype, device=inputs.device)

        outputs = torch.zeros(inputs.size(0), self.hidden_size,
                              dtype=inputs.dtype, device=inputs.device)

        begin = 0
        for batch in batch_sizes:
            new_hx = self.rnn(
                inputs[begin:begin + batch],
                hx[0:batch],
                att_scores[begin:begin + batch])
            outputs[begin:begin + batch] = new_hx
            hx = new_hx
            begin += batch
        return PackedSequence(outputs, batch_sizes, sorted_indices, unsorted_indices)


if __name__ == '__main__':
    pass
    # seq_value_len_list = (torch.randn((128, 20, 32)), torch.randn((128, 1)))
    # # 支持mode=['max','mean,'sum']
    # # 支持按照一定需求进行masking操作
    # seq_pooling_layer = SequencePoolingLayer(mode='sum', supports_masking=True, device='cpu')
    # print(seq_pooling_layer(seq_value_len_list).shape)  # (128, 1, 32)

    # batch_size=128
    # embedding_size=4
    # mask = torch.BoolTensor([batch_size, 20])
    # query = torch.randn((batch_size, 1, embedding_size))
    # keys = torch.randn((batch_size, 20, embedding_size))
    # keys_length = torch.randn(batch_size, 1)
    # # 如果supports_masking=True, mask需要为torch.BoolTensor([B, T])
    # # 如果supports_masking=False，mask可以为None
    # attention_pooling_layer = AttentionSequencePoolingLayer(att_hidden_units=(80, 40), att_activation='sigmoid', weight_normalization=False,
    #                                                         return_score=False, supports_masking=False, embedding_dim=4)
    # print(attention_pooling_layer(query, keys, keys_length, mask=None).shape)  # return_score=False: (128, 2, 4), return_score=True:(128, 2, 20)



