#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   nfm.py
@Time    :   2022/04/23 20:01:10
@Author  :   weiguang 
'''
from turtle import forward
from colorama import init
import torch
import torch.nn as nn

from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.inputs import combined_dnn_input
from deepctr_torch.layers import DNN, BiInteractionPooling

class NFM(BaseModel):
    '''
    NFM Network
    Reference:
        [1] He X, Chua T S. Neural factorization machines for sparse predictive analytics[C]//Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval. ACM, 2017: 355-364. (https://arxiv.org/abs/1708.05027)
    '''
    def __init__(self, linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(128, 128),
                 l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_dnn=0, init_std=0.0001, seed=1024, bi_dropout=0,
                 dnn_dropout=0, dnn_activation='relu', task='binary', device='cpu', gpus=None):
        """
        Args:
            linear_feature_columns (_type_): 线性特征列
            dnn_feature_columns (_type_): dnn特征
            dnn_hidden_units (tuple, optional): dnn隐藏层. Defaults to (128, 128).
            l2_reg_embedding (_type_, optional): l2-embedding正则. Defaults to 1e-5.
            l2_reg_linear (_type_, optional): linear线性模型. Defaults to 1e-5.
            l2_reg_dnn (int, optional): dnn正则. Defaults to 0.
            init_std (float, optional): 初始化矩阵. Defaults to 0.0001.
            seed (int, optional): 种子. Defaults to 1024.
            bi_dropout (int, optional): bi-dropout值. Defaults to 0.
            dnn_dropout (int, optional): dnn-dropout值. Defaults to 0.
            dnn_activation (str, optional): dnn激活函数. Defaults to 'relu'.
            task (str, optional): 任务. Defaults to 'binary'.
            device (str, optional): 计算类型. Defaults to 'cpu'.
            gpus (_type_, optional): GPUS个数. Defaults to None.
        """
        super(NFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                  l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                  device=device, gpus=gpus)
        # DNN神经网络
        self.dnn = DNN(self.compute_input_dim(dnn_feature_columns, include_sparse=False)+self.embedding_size, 
                       dnn_hidden_units, activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                       use_bn=False, init_std=init_std, device=device)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)

        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        # 将领域特征进行Pooling
        self.bi_pooling = BiInteractionPooling()
        self.bi_dropout = bi_dropout
        if self.bi_dropout > 0:
            self.dropout = nn.Dropout(bi_dropout)
        self.to(device)
    
    def forward(self, X):
        # 将输入X分为sparse_embedding_list, dense_value_list
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        # 线性模型
        linear_logit = self.linear_model(X)
        # 输入参数
        fm_input = torch.cat(sparse_embedding_list, dim=1)
        bi_out = self.bi_pooling(fm_input)
        if self.bi_dropout:
            bi_out = self.dropout(bi_out)
        # 将Pooling与dense_value_list组合
        dnn_input = combined_dnn_input([bi_out], dense_value_list)
        # DNN模型
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)

        logit = linear_logit + dnn_logit
        # 输出特征
        y_pred = self.out(logit)
        return y_pred

