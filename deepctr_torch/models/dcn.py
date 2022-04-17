#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   dcn.py
@Time    :   2022/04/17 17:10:15
@Author  :   weiguang 
'''
from turtle import forward
import torch
import torch.nn as nn
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.inputs import combined_dnn_input
from deepctr_torch.layers import CrossNet, DNN

class DCN(BaseModel):
    '''
    Deep Cross网络
    Reference:
        [1] Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12. (https://arxiv.org/abs/1708.05123)
        [2] Wang R, Shivanna R, Cheng D Z, et al. DCN-M: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems[J]. 2020. (https://arxiv.org/abs/2008.13535)
    '''
    def __init__(self, linear_feature_columns, dnn_feature_columns, cross_num=2, cross_parameterization='vector',
                 dnn_hidden_units=(128, 128), l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_cross=0.00001,
                 l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False,
                 task='binary', device='cpu', gpus=None):
        super(DCN, self).__init__(linear_feature_columns=linear_feature_columns,
                                  dnn_feature_columns=dnn_feature_columns, l2_reg_embedding=l2_reg_embedding,
                                  init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)
        self.dnn_hidden_units = dnn_hidden_units
        self.cross_num = cross_num
        # DNN网络
        self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units, activation=dnn_activation,
                       use_bn=dnn_use_bn, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, init_std=init_std, device=device)
        # dnn与cross
        if len(self.dnn_hidden_units) > 0 and self.cross_num > 0:
            dnn_linear_in_feature = self.compute_input_dim(dnn_feature_columns) + dnn_hidden_units[-1]
        # 仅dnn
        elif len(self.dnn_hidden_units) > 0:
            dnn_linear_in_feature = dnn_hidden_units[-1]
        # 仅cross
        elif self.cross_num > 0:
            dnn_linear_in_feature = self.compute_input_dim(dnn_feature_columns)
        # dnn线性模型与crossnet
        self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1, bias=False).to(device)
        self.crossnet = CrossNet(in_features=self.compute_input_dim(dnn_feature_columns), layer_num=cross_num, 
                                 parameterization=cross_parameterization, device=device)
        # 添加正则项
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn
        )
        self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_linear)
        self.add_regularization_weight(self.crossnet.kernels, l2=l2_reg_cross)
        self.to(device)
    
    def forward(self, X):
        # 线性模型
        logit = self.linear_model(X)
        # dense与sparse分析
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        # dnn & cross
        if len(self.dnn_hidden_units) > 0 and self.cross_num > 0:
            deep_out = self.dnn(dnn_input)
            cross_out = self.crossnet(dnn_input)
            stack_out = torch.cat((cross_out, deep_out), dim=-1)
            logit += self.dnn_linear(stack_out)
        # dnn
        elif len(self.dnn_hidden_units) > 0:
            deep_out = self.dnn(dnn_input)
            logit += self.dnn_linear(deep_out)
        # cross
        elif self.cross_num > 0:
            cross_out = self.crossnet(dnn_input)
            logit += self.dnn_linear(cross_out)
        else:
            pass
        # 通过sigmoid输出结果
        y_pred = self.out(logit)
        return y_pred

