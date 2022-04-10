#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   afm.py
@Time    :   2022/04/10 18:39:32
@Author  :   weiguang 
'''

"""
Reference:
    [1] Xiao J, Ye H, He X, et al. Attentional factorization machines: Learning the weight of feature interactions via attention networks[J]. arXiv preprint arXiv:1708.04617, 2017.
    (https://arxiv.org/abs/1708.04617)
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from basemodel import BaseModel
from layers import FM, AFMLayer
from inputs import *

class AFM(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, use_attention=True, attention_factor=8,
                l2_reg_linear=1e-5, l2_reg_embedding=1e-5, l2_reg_att=1e-5, afm_dropout=0, init_std=0.0001, seed=1024,
                task='binary', device='cpu', gpus=None):
        """
        Attention + FM
        Args:
            linear_feature_columns (_type_): 线性部分特征
            dnn_feature_columns (_type_): FM部分特征
            use_attention (bool, optional): 是否采用attention,不采用即为FM. Defaults to True.
            attention_factor (int, optional): attention网络输出维数,(embedding_size, attention_factor). Defaults to 8.
            l2_reg_linear (_type_, optional): 线性网络正则. Defaults to 1e-5.
            l2_reg_embedding (_type_, optional): embedding正则. Defaults to 1e-5.
            l2_reg_att (_type_, optional): attention正则. Defaults to 1e-5.
            afm_dropout (int, optional): afm中dropout值. Defaults to 0.
            init_std (float, optional): 初始化std. Defaults to 0.0001.
            seed (int, optional): 种子. Defaults to 1024.
            task (str, optional): task类型. Defaults to 'binary'.
            device (str, optional): 设备类型. Defaults to 'cpu'.
            gpus (_type_, optional): gpus数量. Defaults to None.
        """
        super(AFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                  l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                  device=device, gpus=gpus)
        self.use_attention = use_attention
        # 如果使用attention，则采用AFMLayer
        # 否则采用FM
        if use_attention:
            self.fm = AFMLayer(self.embedding_size, attention_factor, l2_reg_att, afm_dropout,
                               seed, device)
            self.add_regularization_weight(self.fm.attention_W, l2=l2_reg_att)
        else:
            self.fm = FM()
        self.to(device)
    
    def forward(self, X):
        sparse_embedding_list, _ = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                   self.embedding_dict, support_dense=False)
        logit = self.linear_model(X)
        if len(sparse_embedding_list) > 0:
            # 如果采用attention，直接输入sparse_embedding_list
            # 否则将sparse_embedding_list按照dim=1进行合并
            if self.use_attention:
                logit += self.fm(sparse_embedding_list)
            else:
                logit += self.fm(torch.cat(sparse_embedding_list, dim=1))
        y_pred = self.out(logit)
        return y_pred
