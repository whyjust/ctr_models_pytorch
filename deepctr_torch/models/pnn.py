#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   pnn.py
@Time    :   2022/04/23 22:29:43
@Author  :   weiguang 
'''
import torch
import torch.nn as nn
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.inputs import combined_dnn_input
from deepctr_torch.layers import DNN, concat_fun, InnerProductLayer, OutterProductLayer

class PNN(BaseModel):
    '''
    Product-based Nerwork
    Reference:
        [1] Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response prediction[C]//Data Mining (ICDM), 2016 IEEE 16th International Conference on. IEEE, 2016: 1149-1154.(https://arxiv.org/pdf/1611.00144.pdf)
    '''
    def __init__(self, dnn_feature_columns, dnn_hidden_units=(128, 128), l2_reg_embedding=1e-5, l2_reg_dnn=0,
                 init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', use_inner=True, use_outter=False,
                 kernel_type='mat', task='binary', device='cpu', gpus=None):
        super(PNN, self).__init__([], dnn_feature_columns, l2_reg_linear=0, l2_reg_embedding=l2_reg_embedding,
                                  init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)

        if kernel_type not in ['mat', 'vec', 'num']:
            raise ValueError("kernel_type must be mat,vec or num")
        
        self.use_inner = use_inner
        self.use_outter = use_outter
        self.kernel_type = kernel_type
        self.task = task

        product_out_dim = 0
        # 计算输入维度
        num_inputs = self.compute_input_dim(dnn_feature_columns, include_dense=False, feature_group=True)
        num_pairs = int(num_inputs * (num_inputs -1) / 2)

        if self.use_inner:
            product_out_dim += num_pairs
            self.innerproduct = InnerProductLayer(device=device)
        
        if self.use_outter:
            product_out_dim += num_pairs
            self.outterproduct = OutterProductLayer(num_inputs, self.embedding_size, \
                kernel_type=kernel_type, device=device)
        # dnn深度网络
        self.dnn = DNN(product_out_dim + self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                       activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=False,
                       init_std=init_std, device=device)
        # 线性模型
        self.dnn_linear = nn.Linear(
            dnn_hidden_units[-1], 1, bias=False
        ).to(device)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

        self.to(device)

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        # 将sparse_embedding_list合并+扁平化
        linear_signal = torch.flatten(concat_fun(sparse_embedding_list), start_dim=1)
        # InnerProductLayer
        if self.use_inner:
            inner_product = torch.flatten(
                self.innerproduct(sparse_embedding_list), start_dim=1
            )
        # OutterProductLayer
        if self.use_outter:
            outer_product = self.outterproduct(
                sparse_embedding_list
            )
        # InnerProductLayer & OutterProductLayer
        if self.use_outter and self.use_inner:
            product_layer = torch.cat(
                [linear_signal, inner_product, outer_product], dim=1
            )
        elif self.use_outter:
            product_layer = torch.cat([
                linear_signal, outer_product
            ], dim=1)
        elif self.use_inner:
            product_layer = torch.cat([
                linear_signal, inner_product
            ], dim=1)
        else:
            product_layer = linear_signal
        # DNN输入
        dnn_input = combined_dnn_input([product_layer], dense_value_list)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)
        logit = dnn_logit

        y_pred = self.out(logit)
        return y_pred

