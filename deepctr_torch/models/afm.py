# -*- coding:utf-8 -*-
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
