#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   afn.py
@Time    :   2022/04/16 21:18:33
@Author  :   weiguang 
'''
import torch
import torch.nn as nn
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.layers import LogTransformLayer, DNN

class AFN(BaseModel):
    """
    自适应的因子分解机
    Reference:
    [1] Cheng, W., Shen, Y. and Huang, L. 2020. Adaptive Factorization Network: Learning Adaptive-Order Feature
         Interactions. Proceedings of the AAAI Conference on Artificial Intelligence. 34, 04 (Apr. 2020), 3609-3616.
    """
    def __init__(self, linear_feature_columns, dnn_feature_columns,
                 ltl_hidden_size=256, afn_dnn_hidden_units=(256, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0,
                 init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu',
                 task='binary', device='cpu', gpus=None):
        """
        Args:
            linear_feature_columns (_type_): 线性特征
            dnn_feature_columns (_type_): dnn网络特征
            ltl_hidden_size (int, optional): ltl网络隐层. Defaults to 256.
            afn_dnn_hidden_units (tuple, optional): afn-dnn中隐层神经元. Defaults to (256, 128).
            l2_reg_linear (float, optional): 线性正则. Defaults to 0.00001.
            l2_reg_embedding (float, optional): embedding正则. Defaults to 0.00001.
            l2_reg_dnn (int, optional): dnn正则. Defaults to 0.
            init_std (float, optional): 初始化参数. Defaults to 0.0001.
            seed (int, optional): 种子. Defaults to 1024.
            dnn_dropout (int, optional): dnn. Defaults to 0.
            dnn_activation (str, optional): dnn激活函数. Defaults to 'relu'.
            task (str, optional): 任务. Defaults to 'binary'.
            device (str, optional): 计算类型. Defaults to 'cpu'.
            gpus (_type_, optional): GPUs数量. Defaults to None.
        """

        super(AFN, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                  l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                  device=device, gpus=gpus)
        # 自适应因子分解网络对数变化，任意阶交叉特征
        self.ltl = LogTransformLayer(len(self.embedding_dict), self.embedding_size, ltl_hidden_size)
        # DNN深度网络
        self.afn_dnn = DNN(self.embedding_size * ltl_hidden_size, afn_dnn_hidden_units,
                       activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=True,
                       init_std=init_std, device=device)
        # afn_dnn线性转换
        self.afn_dnn_linear = nn.Linear(afn_dnn_hidden_units[-1], 1)
        self.to(device)
    
    def forward(self, X):
        sparse_embedding_list, _ = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                   self.embedding_dict)
        logit = self.linear_model(X)
        if len(sparse_embedding_list) == 0:
            raise ValueError('Sparse embeddings not provided. AFN only accepts sparse embeddings as input.')
        # 将特征按照dim=1合并
        afn_input = torch.cat(sparse_embedding_list, dim=1)
        # 因子分解机，对数任意阶交叉特征
        ltl_result = self.ltl(afn_input)
        # DNN深度网络
        afn_logit = self.afn_dnn(ltl_result)
        # afn_dnn线性转换
        afn_logit = self.afn_dnn_linear(afn_logit)
        
        logit += afn_logit
        y_pred = self.out(logit)
        return y_pred


