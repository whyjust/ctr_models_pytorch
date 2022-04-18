#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ccpm.py
@Time    :   2022/04/17 15:55:00
@Author  :   weiguang 
'''
from turtle import forward
import torch
import torch.nn as nn

from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.layers.core import DNN
from deepctr_torch.layers.interaction import ConvLayer
from deepctr_torch.layers.utils import concat_fun

class CCPM(BaseModel):
    """
    CCPM: Instantiates the Convolutional Click Prediction Model architecture
    Reference:
        [1] Liu Q, Yu F, Wu S, et al. A convolutional click prediction model[C]//Proceedings of the 24th ACM International on Conference on Information and Knowledge Management. ACM, 2015: 1743-1746.
        (http://ir.ia.ac.cn/bitstream/173211/12337/1/A%20Convolutional%20Click%20Prediction%20Model.pdf)
    """
    def __init__(self, linear_feature_columns, dnn_feature_columns, conv_kernel_width=(6, 5), conv_filters=(4, 4),
                dnn_hidden_units=(256, ), l2_reg_linear=1e-5, l2_reg_embedding=1e-5, l2_reg_dnn=0, dnn_dropout=0, 
                init_std=0.0001, seed=1024, task='binary', device='cpu', dnn_use_bn=False, dnn_activation='relu', gpus=None):
        super(CCPM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                   l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                   device=device, gpus=gpus)
        if len(conv_kernel_width) != len(conv_filters):
            raise ValueError(
                "conv_kernel_width must have same element with conv_filters")

        # 将sparse与dense合并
        field_size = self.compute_input_dim(dnn_feature_columns, include_dense=False, feature_group=True)
        # CCPM中卷积核
        self.conv_layer = ConvLayer(field_size=field_size, conv_kernel_width=conv_kernel_width,
                                    conv_filters=conv_filters, device=device)
        # dnn输入维度
        self.dnn_input_dim = self.conv_layer.field_shape * self.embedding_size * conv_filters[-1]
        # DNN网络
        self.dnn = DNN(self.dnn_input_dim, dnn_hidden_units, activation=dnn_hidden_units, l2_reg=l2_reg_dnn, 
                       dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std, device=device)
        # dnn线性模型
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn
        )
        self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        self.to(device)

    def forward(self, X):
        linear_logit = self.linear_model(X)
        sparse_embedding_list, _ = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict, support_dense=False)
        if len(sparse_embedding_list) == 0:
            raise ValueError("must have the embedding feature,now the embedding feature is None!")

        # sparse_embedding_list按照dim=1进行组合
        conv_input = concat_fun(sparse_embedding_list, axis=1)
        # 扩增conv_input维度
        conv_input_concat = torch.unsqueeze(conv_input, 1)
        # 卷积核pooling
        pooling_result = self.conv_layer(conv_input_concat)
        # 将pooling数据拉平
        flatten_result = pooling_result.view(pooling_result.size(0), -1)
        # DNN模型输出
        dnn_output = self.dnn(flatten_result)
        # dnn输出结果
        dnn_logit = self.dnn_linear(dnn_output)
        # 将线性与dnn结果合并
        logit = linear_logit + dnn_logit
        y_pred = self.out(logit)
        return y_pred


