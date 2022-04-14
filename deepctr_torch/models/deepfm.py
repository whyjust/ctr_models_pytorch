#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   deepfm.py
@Time    :   2022/04/10 18:39:07
@Author  :   weiguang 
'''
from turtle import forward
import torch
import torch.nn as nn
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.inputs import combined_dnn_input
from deepctr_torch.layers import FM, DNN, activation

class DeepFM(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, use_fm=True,
                 dnn_hidden_units=(256, 128), l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None):
        """
        DNN + FM
        Reference:
            [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)
        Args:
            linear_feature_columns (_type_): 线性特征列
            dnn_feature_columns (_type_): dnn特征列
            use_fm (bool, optional): 是否使用FM. Defaults to True.
            dnn_hidden_units (tuple, optional): dnn隐藏层神经元组. Defaults to (256, 128).
            l2_reg_linear (float, optional): 线性正则. Defaults to 0.00001.
            l2_reg_embedding (float, optional): embedding正则. Defaults to 0.00001.
            l2_reg_dnn (int, optional): dnn正则. Defaults to 0.
            init_std (float, optional): 初始化std. Defaults to 0.0001.
            seed (int, optional): 种子. Defaults to 1024.
            dnn_dropout (int, optional): dnn-dropout比例. Defaults to 0.
            dnn_activation (str, optional): dnn-激活函数. Defaults to 'relu'.
            dnn_use_bn (bool, optional): dnn是否加入batchNormalize. Defaults to False.
            task (str, optional): 分类任务. Defaults to 'binary'.
            device (str, optional): 计算类型. Defaults to 'cpu'.
            gpus (_type_, optional): GPU个数. Defaults to None.
        """
        super(DeepFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus)
        
        self.use_fm = use_fm
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        if use_fm:
            self.fm = FM()
        
        if self.use_dnn:
            # DNN模型训练
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            # 对结果线性转换输出 
            self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
            # dnn对应weight添加l2_reg_dnn参数
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn
            )
            # 线性weight添加l2_reg_dnn
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

        self.to(device)
    
    def forward(self, X):
        # 将输入X分别对应sparseFeat,VarLenSparseFeat,denseFeat进行处理输出
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, 
                                                                                  self.embedding_dict, support_dense=True)
        # 对上述特征列进行linear转换
        logit = self.linear_model(X)

        if self.use_fm and len(sparse_embedding_list) > 0:
            # 将sparse-embedding后的向量在dim=1上组合
            fm_input = torch.cat(sparse_embedding_list, dim=1)
            # 输出fm的结果
            logit += self.fm(fm_input)
        
        if self.use_dnn:
            # dnn输出为sparse-embedding向量与dense向量
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            logit += dnn_logit
        # 对应一个predictionLayer，task=binary则为sigmoid映射
        y_pred = self.out(logit)
        return y_pred

