#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   wdl.py
@Time    :   2022/04/14 20:21:42
@Author  :   weiguang 
'''
import torch.nn as nn
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.inputs import combined_dnn_input
from deepctr_torch.layers import DNN
class WDL(BaseModel):
    """
    wide & deep 推荐系统
    Reference:
        [1] Cheng H T, Koc L, Harmsen J, et al. Wide & deep learning for recommender systems[C]//Proceedings of the 1st Workshop on Deep Learning for Recommender Systems. ACM, 2016: 7-10.(https://arxiv.org/pdf/1606.07792.pdf)
    """
    def __init__(self, linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 128),
                 l2_reg_linear=1e-5, l2_reg_embedding=1e-5, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu',
                 dnn_use_bn=False, task='binary', device='cpu', gpus=None):
        """
        Args:
            linear_feature_columns (_type_): 线性部分的所有特征
            dnn_feature_columns (_type_): DNN特征列表
            dnn_hidden_units (tuple, optional): dnn隐层节点. Defaults to (256, 128).
            l2_reg_linear (_type_, optional): 线性L2正则. Defaults to 1e-5.
            l2_reg_embedding (_type_, optional): 稠密L2正则. Defaults to 1e-5.
            l2_reg_dnn (int, optional): DNN正则. Defaults to 0.
            init_std (float, optional): 初始化. Defaults to 0.0001.
            seed (int, optional): 种子. Defaults to 1024.
            dnn_dropout (int, optional): dnn中dropout. Defaults to 0.
            dnn_activation (str, optional): dnn激活函数. Defaults to 'relu'.
            dnn_use_bn (bool, optional): dnn是否采用BN. Defaults to False.
            task (str, optional): 任务名. Defaults to 'binary'.
            device (str, optional): 设备. Defaults to 'cpu'.
            gpus (_type_, optional): GPUS个数. Defaults to None.
        """
        super(WDL, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                  l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                  device=device, gpus=gpus)

        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        # 是否采用dnn
        if self.use_dnn:
            # dnn模型
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units, activation=dnn_activation,
                            l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std, device=device)
            self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
            # 添加正则参数
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters), l2=l2_reg_dnn
            )
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        self.to(device)

    def forward(self, X):
        # 将输入X按照不同类型做不同处理
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        # 线性模型logit结果
        logit = self.linear_model(X)
        if self.use_dnn:
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            # 线性logit+dnn logit
            logit += dnn_logit
        
        # 累加的logit再做一层sigmoid转换
        # 输出概率
        y_pred = self.out(logit)
        return y_pred