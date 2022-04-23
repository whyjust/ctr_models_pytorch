#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   mlr.py
@Time    :   2022/04/20 23:28:23
@Author  :   weiguang 
'''
import torch
import torch.nn as nn
from deepctr_torch.models.basemodel import Linear, BaseModel
from deepctr_torch.inputs import build_input_features
from deepctr_torch.layers import PredictionLayer

class MLR(BaseModel):
    '''
    Mixed Logistic Regression/Piece-wise Linear Model
    Reference:
        [1] Gai K, Zhu X, Li H, et al. Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction[J]. arXiv preprint arXiv:1704.05194, 2017.(https://arxiv.org/abs/1704.05194)
    '''
    def __init__(self, region_feature_columns, base_feature_columns=None, bias_feature_columns=None,
                 region_num=4, l2_reg_linear=1e-5, init_std=0.0001, seed=1024, task='binary', device='cpu', gpus=None):
        super(MLR, self).__init__(region_feature_columns, region_feature_columns, task=task, device=device, gpus=gpus)

        if region_num <= 1:
            raise ValueError("region_num must > 1")
        
        self.l2_reg_linear = l2_reg_linear
        self.init_std = init_std
        self.seed = seed
        self.device = device

        self.region_num = region_num
        self.region_feature_columns = region_feature_columns
        self.base_feature_columns = base_feature_columns
        self.bias_feature_columns = bias_feature_columns

        if base_feature_columns is None or len(base_feature_columns)==0:
            self.base_feature_columns = region_feature_columns
        
        if bias_feature_columns is None:
            self.bias_feature_columns = []

        # 将3个特征列表合并统计
        self.feature_index = build_input_features(self.region_feature_columns + self.base_feature_columns + self.bias_feature_columns)
        # region线性模型
        self.region_linear_model = nn.ModuleList([
            Linear(self.region_feature_columns, self.feature_index, self.init_std, self.device) \
                for i in range(self.region_num)
        ])
        
        # base线性模型
        self.base_linear_model = nn.ModuleList([
            Linear(self.base_feature_columns, self.feature_index, self.init_std, self.device) \
                for i in range(self.region_num)
        ])
        # bias模型包含Linear与PredictionLayer
        if self.bias_feature_columns is not None and len(self.bias_feature_columns) > 0:
            self.bias_model = nn.Sequential(
                Linear(self.bias_feature_columns, self.feature_index, self.init_std, self.device),
                PredictionLayer(task='binary', use_bias=False)
            )
        self.prediction_layer = PredictionLayer(task=task, use_bias=False)
        self.to(self.device)

    def get_regoin_score(self, inputs, region_number):
        region_logit = torch.cat([
            self.region_linear_model[i](inputs) for i in range(region_number)
        ], dim=-1)
        region_score = nn.Softmax(dim=-1)(region_logit)
        return region_score

    def get_learner_score(self, inputs, region_number):
        learner_score = self.prediction_layer(
            torch.cat([self.region_linear_model[i](inputs) for i in range(region_number)], dim=-1)
        )
        return learner_score
    
    def forward(self, X):
        # 获取region score
        region_score = self.get_regoin_score(X, self.region_num)
        # 获取learner score
        learner_score = self.get_learner_score(X, self.region_num)
        # sum(region_score * learner_score)
        final_logit = torch.sum(
            region_score * learner_score, dim=-1, keepdim=True
        )
        if self.bias_feature_columns is not None and len(self.bias_feature_columns) > 0:
            bias_score = self.bias_model(X)
            final_logit = final_logit * bias_score
        return final_logit
    
