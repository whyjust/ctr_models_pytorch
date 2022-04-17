#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   autoint.py
@Time    :   2022/04/16 23:06:01
@Author  :   weiguang 
'''
from inspect import stack
from turtle import forward
import torch
import torch.nn as nn
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.inputs import combined_dnn_input
from deepctr_torch.layers import DNN, concat_fun, InteractingLayer

class AutoInt(BaseModel):
    """
    基于自注意力神经网络的自动特征交互学习
    Reference:
        [1] Song W, Shi C, Xiao Z, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks[J]. arXiv preprint arXiv:1810.11921, 2018.(https://arxiv.org/abs/1810.11921)
    """
    def __init__(self, linear_feature_columns, dnn_feature_columns, att_layer_num=3,
                 att_head_num=2, att_res=True, dnn_hidden_units=(256, 128), dnn_activation='relu',
                 l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=0, init_std=0.0001, seed=1024,
                 task='binary', device='cpu', gpus=None):

        super(AutoInt, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=0,
                                      l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                      device=device, gpus=gpus)
        if len(dnn_hidden_units) <= 0 and att_layer_num <= 0:
            raise ValueError("Either hidden_layer or att_layer_num must > 0")
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        field_num = len(self.embedding_dict)
        embedding_size = len(self.embedding_dict)

        # 如果dnn_hidden_units与att_layer_num大于0
        # dnn_hidden_units -> dnn_hidden_units[-1]
        # dnn_linear_in_feature -> field_num * embedding_size
        if len(dnn_hidden_units) and att_layer_num > 0:
            dnn_linear_in_feature = dnn_hidden_units[-1] + field_num * embedding_size
        elif len(dnn_hidden_units) > 0:
            dnn_linear_in_feature = dnn_hidden_units[-1]
        elif att_layer_num > 0:
            dnn_linear_in_feature = field_num * embedding_size
        else:
            raise NotImplementedError
        
        # 线性模型
        self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1, bias=False).to(device)
        self.dnn_hidden_units = dnn_hidden_units
        self.att_layer_num = att_layer_num

        # 是否采用dnn
        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn
            )
        # 通过多头注意力机制的不同领域的相关性模型
        self.int_layers = nn.ModuleList(
            [InteractingLayer(embedding_size, att_head_num, att_res, device=device) for _ in range(att_layer_num)]
        )
        self.to(device)

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        logit = self.linear_model(X)
        # 将sparse_embedding_list按照dim=1合并
        att_input = concat_fun(sparse_embedding_list, axis=1)

        # 遍历自注意力模型
        for layer in self.int_layers:
            att_input = layer(att_input)
        att_output = torch.flatten(att_input, start_dim=1)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        # Deep & Interacting Layer
        if len(self.dnn_hidden_units) > 0 and self.att_layer_num > 0:
            deep_out = self.dnn(dnn_input)
            stack_out = concat_fun([att_output, deep_out])
            logit += self.dnn_linear(stack_out)
        # only deep
        elif len(self.dnn_hidden_units) > 0:
            deep_out = self.dnn(dnn_input)
            logit += self.dnn_linear(deep_out)
        # Only Interacting Layer
        elif self.att_layer_num > 0:
            logit += self.dnn_linear(att_output)
        else:
            pass

        y_pred = self.out(logit)
        return y_pred
        