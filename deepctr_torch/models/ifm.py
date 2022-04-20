#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ifm.py
@Time    :   2022/04/20 22:52:20
@Author  :   weiguang 
'''
from msilib.schema import Class
import torch
import torch.nn as nn
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.inputs import combined_dnn_input, SparseFeat, VarLenSparseFeat
from deepctr_torch.layers import FM, DNN

class IFM(BaseModel):
    '''
    IFM网络
    Reference:
        [1] Yu Y, Wang Z, Yuan B. An Input-aware Factorization Machine for Sparse Prediction[C]//IJCAI. 2019: 1466-1472.(https://www.ijcai.org/Proceedings/2019/0203.pdf)
    '''
    def __init__(self, linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None):
        """
        Args:
            linear_feature_columns (_type_): 线性变量
            dnn_feature_columns (_type_): dnn特征变量
            dnn_hidden_units (tuple, optional): dnn隐层神经元. Defaults to (256, 128).
            l2_reg_linear (float, optional): l2线性正则. Defaults to 0.00001.
            l2_reg_embedding (float, optional): embedding正则. Defaults to 0.00001.
            l2_reg_dnn (int, optional): dnn正则. Defaults to 0.
            init_std (float, optional): 初始化矩阵. Defaults to 0.0001.
            seed (int, optional): 种子. Defaults to 1024.
            dnn_dropout (int, optional): dnn-dropout. Defaults to 0.
            dnn_activation (str, optional): dnn激活函数. Defaults to 'relu'.
            dnn_use_bn (bool, optional): dnn采用BN. Defaults to False.
            task (str, optional): 任务类型. Defaults to 'binary'.
            device (str, optional): 计算类型. Defaults to 'cpu'.
            gpus (_type_, optional): GPUS. Defaults to None.
        """
        super(IFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                  l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                  device=device, gpus=gpus)
        
        if not len(dnn_hidden_units) > 0:
            raise ValueError("dnn_hidden_units is null!")

        # FM因子分子机
        self.fm = FM()
        # DNN神经网络
        self.factor_estimating_net = DNN(self.compute_input_dim(dnn_feature_columns, include_dense=False),
                                         dnn_hidden_units, activation=dnn_activation, l2_reg=l2_reg_dnn, 
                                         dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std, device=device)
        # dnn特征过滤出Sparse与VarLenSparse特征
        self.sparse_feat_num = len(list(
            filter(lambda x: isinstance(x, SparseFeat) or isinstance(x, VarLenSparseFeat), dnn_feature_columns)
        ))
        # 线性model, 输入特征维度dnn_hidden_units[-1], 输出维度sparse_feat_num
        self.transform_weight_matrix_P = nn.Linear(
            dnn_hidden_units[-1], self.sparse_feat_num, bias=False
        ).to(device)
        # 正则weight
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.factor_estimating_net.named_parameters()),
            l2=l2_reg_dnn
        )
        # 正则化weight
        self.add_regularization_weight(self.transform_weight_matrix_P.weight, l2=l2_reg_dnn)
        self.to(device)

    def forward(self, X):
        # 将输入特征进行注册与分类
        sparse_embedding_list, _ = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)

        if not len(sparse_embedding_list) > 0:
            raise ValueError("there are no sparse features")
        # (batch_size, feat_num * embedding_size)
        dnn_input = combined_dnn_input(sparse_embedding_list, [])
        # DNN神经网络
        dnn_output = self.factor_estimating_net(dnn_input)
        # DNN+线性输出: m'_{x}
        dnn_output = self.transform_weight_matrix_P(dnn_output)
        # m_{x,i}
        input_aware_factor = self.sparse_feat_num * dnn_output.softmax(1)

        # 线性模型
        logit = self.linear_model(X, sparse_feat_refine_weight=input_aware_factor)
        fm_input = torch.cat(sparse_embedding_list, dim=1)
        refined_fm_input = fm_input * input_aware_factor.unsqueeze(-1)
        logit += self.fm(refined_fm_input)

        y_pred = self.out(logit)
        return y_pred

    