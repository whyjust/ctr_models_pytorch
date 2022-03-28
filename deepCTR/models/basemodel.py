#!/usr/bin/python
# -*- coding:utf-8 -*-
# @FileName  : activation.py
# @Time      : 2022/3/7 15:20
# @Author    : weiguang

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Data
from sklearn.metrics import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorflow.python.keras.callbacks import CallbackList

from deepCTR.layers import sequence

from ..inputs import build_input_features,SparseFeat, DenseFeat, VarLenSparseFeat,\
    get_varlen_pooling_list, create_embedding_matrix, varlen_embedding_lookup
from ..layers import PredictionLayer
from ..layers.utils import slice_arrays
from ..callbacks import History

class Linear(nn.Module):
    """
    线性模型
    Args:
        nn (_type_): _description_
    """
    def __init__(self, feature_columns, feature_index, init_std=0.0001, device='cpu'):
        """
        Args:
            feature_columns (_type_): 注册过的特征列
            feature_index (_type_): 特征对应的index
            init_std (float, optional): normal初始化. Defaults to 0.0001.
            device (str, optional): 设备. Defaults to 'cpu'.
        """
        super(Linear, self).__init__()
        self.feature_index = feature_index
        self.device = device
        # 将feature_columns中SparseFeat取出来
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)
        ) if len(feature_columns) else []
        # 将feature_columns中DenseFeat取出来
        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)
        ) if len(feature_columns) else []
        # 将feature_columns中VarLenSparseFeat取出来
        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)
        ) if len(feature_columns) else []
        self.embedding_dict = create_embedding_matrix(feature_columns, init_std, linear=True, sparse=False, device=device)
        #         nn.ModuleDict(
        #             {feat.embedding_name: nn.Embedding(feat.dimension, 1, sparse=True) for feat in
        #              self.sparse_feature_columns}
        #         )
        # .to("cuda:1")

        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)
        
        if len(self.dense_feature_columns) > 0:
            self.weight = nn.Parameter(torch.Tensor(sum(fc.dimension for fc in self.dense_feature_columns), 1).to(device))
            torch.nn.init.normal_(self.weight, mean=0, std=init_std)
    
    def forward(self, X, sparse_feat_refine_weight=None):
        """
        Args:
            X (_type_): 特征
            sparse_feat_refine_weight (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0] : self.feature_index[feat.name][1]].long())
            for feat in self.sparse_feature_columns
        ]
        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] 
            for feat in self.dense_feature_columns
        ]
        # 根据lookup查找embedding
        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, 
                                                    self.feature_index, self.varlen_sparse_feature_columns)
        # 将embedding_dict中某个特征进行pooling
        varlen_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                    self.varlen_sparse_feature_columns, self.device)
        sparse_embedding_list += varlen_embedding_list

        linear_logit = torch.zeros([X.shape[0], 1]).to(sparse_embedding_list[0].device)
        if len(sparse_embedding_list) > 0:
            sparse_embedding_cat = torch.cat(sparse_embedding_list, dim=-1)
            if sparse_feat_refine_weight is not None:
                # w_{x,i}=m_{x,i} * w_i (in IFM and DIFM)
                sparse_embedding_cat = sparse_embedding_cat * sparse_feat_refine_weight
            sparse_feat_logit = torch.sum(sparse_embedding_cat, dim=-1, keepdim=False)
            linear_logit += sparse_feat_logit

        if len(dense_value_list) > 0:
            dense_value_logit = torch.cat(dense_value_list, dim=-1).matmul(self.weight)
            linear_logit += dense_value_logit
        return linear_logit


