#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   xdeepfm.py
@Time    :   2022/04/25 21:49:08
@Author  :   weiguang 
'''
from turtle import forward
import torch
import torch.nn as nn
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.inputs import combined_dnn_input
from deepctr_torch.layers import DNN, CIN

class xDeepFM(BaseModel):
    '''
    XDeepFM
    Reference:
        [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)
    '''
    def __init__(self, linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 256),
                 cin_layer_size=(256, 128,), cin_split_half=True, cin_activation='relu', l2_reg_linear=0.00001,
                 l2_reg_embedding=0.00001, l2_reg_dnn=0, l2_reg_cin=0, init_std=0.0001, seed=1024, dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None):
        """
        Args:
            linear_feature_columns (_type_): 线性模型特征
            dnn_feature_columns (_type_): dnn特征列
            dnn_hidden_units (tuple, optional): dnn隐藏层. Defaults to (256, 256).
            cin_layer_size (tuple, optional): cin-layer大小. Defaults to (256, 128,).
            cin_split_half (bool, optional): 如果设置为True,每个隐藏的特征图的一半将连接到输出单元. Defaults to True.
            cin_activation (str, optional): cin激活函数. Defaults to 'relu'.
            l2_reg_linear (float, optional): 线性正则. Defaults to 0.00001.
            l2_reg_embedding (float, optional): embedding正则. Defaults to 0.00001.
            l2_reg_dnn (int, optional): dnn正则. Defaults to 0.
            l2_reg_cin (int, optional): cin正则. Defaults to 0.
            init_std (float, optional): 初始化矩阵. Defaults to 0.0001.
            seed (int, optional): 种子. Defaults to 1024.
            dnn_dropout (int, optional): dnn-dropout. Defaults to 0.
            dnn_activation (str, optional): dnn激活函数. Defaults to 'relu'.
            dnn_use_bn (bool, optional): dnn采用BN. Defaults to False.
            task (str, optional): 分类. Defaults to 'binary'.
            device (str, optional): 计算类型. Defaults to 'cpu'.
            gpus (_type_, optional): GPUS. Defaults to None.
        """
        super(xDeepFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                      l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                      device=device, gpus=gpus)
        self.dnn_hidden_units = dnn_hidden_units
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        if self.use_dnn:
            # DNN神经网络
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units, activation=dnn_activation,
                           l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std, device=device)
            # Linear线性模型
            self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters(), l2=l2_reg_dnn)
            )
            self.add_regularization_weight(
                self.dnn_linear.weight, l2=l2_reg_dnn
            )

        self.cin_layer_size = cin_layer_size
        self.use_cin = len(self.cin_layer_size) > 0 and len(dnn_feature_columns) > 0
        
        # 采用CIN layer
        if self.use_cin:
            field_num = len(self.embedding_dict)
            # 是否为偶数
            if cin_split_half == True:
                self.featuremap_num = sum(cin_layer_size[:-1]) // 2 + cin_layer_size[-1]
            else:
                self.featuremap_num = sum(cin_layer_size)
            # CIN layer
            self.cin = CIN(field_num, cin_layer_size, cin_activation, cin_split_half, l2_reg_cin, seed, device=device)
            # 线性模型
            self.cin_linear = nn.Linear(self.featuremap_num, 1, bias=False).to(device)
            # 添加正则
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0], self.cin.named_parameters()), l2=l2_reg_cin
            )
        self.to(device)

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        # 线性layer
        linear_logit = self.linear_model(X)
        # 采用cin layer
        if self.use_cin:
            cin_input = torch.cat(sparse_embedding_list, dim=1)
            cin_output = self.cin(cin_input)
            cin_logit = self.cin_linear(cin_output)
        # 采用dnn layer
        if self.use_dnn:
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)

        # only linear
        if len(self.dnn_hidden_units) == 0 and len(self.cin_layer_size) == 0:
            final_logit = linear_logit
        # linear + cin
        elif len(self.dnn_hidden_units) == 0 and len(self.cin_layer_size) > 0:
            final_logit = linear_logit + cin_logit
        # dnn + linear
        elif len(self.dnn_hidden_units) > 0 and len(self.cin_layer_size) == 0:
            final_logit = linear_logit + dnn_logit
        # linear + dnn + cin
        elif len(self.dnn_hidden_units) > 0 and len(self.cin_layer_size) > 0:
            final_logit = linear_logit + dnn_logit + cin_logit
        else:
            raise NotImplementedError
        
        y_pred = self.out(final_logit)
        return y_pred
    