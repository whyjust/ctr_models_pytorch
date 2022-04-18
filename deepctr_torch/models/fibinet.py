#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   fibinet.py
@Time    :   2022/04/18 23:09:32
@Author  :   weiguang 
'''
from turtle import forward
import torch
import torch.nn as nn
from deepctr_torch.layers.interaction import BilinearInteraction
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.inputs import combined_dnn_input, SparseFeat, DenseFeat, VarLenSparseFeat
from deepctr_torch.layers import SENETLayer, BiInteractionPooling, DNN

class FiBiNET(BaseModel):
    '''
    FiBiNET: Instantiates the Feature Importance and Bilinear feature Interaction NETwork architecture
    Reference:
        [1] Huang T, Zhang Z, Zhang J. FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1905.09433, 2019.
    '''
    def __init__(self, linear_feature_columns, dnn_feature_columns, bilinear_type='interaction',
                 reduction_ratio=3, dnn_hidden_units=(128, 128), l2_reg_linear=1e-5,
                 l2_reg_embedding=1e-5, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu',
                 task='binary', device='cpu', gpus=None):
        """
        Args:
            linear_feature_columns (_type_): 线性模型的特征列
            dnn_feature_columns (_type_): DNN模型的特征列
            bilinear_type (str, optional): Bilinear Interaction Layer中双线性模型. Defaults to 'interaction'.
            reduction_ratio (int, optional): SENET layer中缩小的比例. Defaults to 3.
            dnn_hidden_units (tuple, optional): dnn隐层. Defaults to (128, 128).
            l2_reg_linear (_type_, optional): 线性正则. Defaults to 1e-5.
            l2_reg_embedding (_type_, optional): embedding正则. Defaults to 1e-5.
            l2_reg_dnn (int, optional): dnn正则. Defaults to 0.
            init_std (float, optional): _description_. Defaults to 0.0001.
            seed (int, optional): _description_. Defaults to 1024.
            dnn_dropout (int, optional): _description_. Defaults to 0.
            dnn_activation (str, optional): _description_. Defaults to 'relu'.
            task (str, optional): _description_. Defaults to 'binary'.
            device (str, optional): _description_. Defaults to 'cpu'.
            gpus (_type_, optional): _description_. Defaults to None.
        """
        super(FiBiNET, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                      l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                      device=device, gpus=gpus)
        # 线性特征
        self.linear_feature_columns = linear_feature_columns
        # dnn特征
        self.dnn_feature_columns = dnn_feature_columns
        # 领域size
        self.field_size = len(self.embedding_dict)
        # SENETLayer层
        self.SE = SENETLayer(self.field_size, reduction_ratio, seed, device)
        # bilinear_type->all、each、interaction
        self.Bilinear = BilinearInteraction(self.field_size, self.embedding_size, bilinear_type, seed, device)
        # DNN网络层
        self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units, activation=dnn_activation, 
                       l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=False, init_std=init_std, device=device)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)

    def compute_input_dim(self, feature_columns, include_sparse=True, include_dense=True):
        # 过滤出sparse与dense特征列
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)
        ) if len(feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)
        ) if len(feature_columns) else []
        field_size = len(sparse_feature_columns)
        # 累加维度
        dense_input_dim = sum(map(lambda x: x.dimension, dense_feature_columns))
        embedding_size = sparse_feature_columns[0].embedding_dim
        sparse_input_dim = field_size * (field_size - 1) * embedding_size
        # 返回输入dim
        input_dim = 0
        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim
        return input_dim

    
    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        # sparse特征合并
        sparse_embedding_input = torch.cat(sparse_embedding_list, dim=1)
        senet_output = self.SE(sparse_embedding_input)
        senet_bilinear_out = self.Bilinear(senet_output)
        bilinear_out = self.Bilinear(sparse_embedding_input)

        # 线性模型
        linear_logit = self.linear_model(X)
        temp = torch.split(torch.cat((senet_bilinear_out, bilinear_out), dim=1), 1, dim=1)
        # 计算dnn模型维度
        dnn_input = combined_dnn_input(temp, dense_value_list)
        # dnn网络
        dnn_output = self.dnn(dnn_input)
        # 线性模型
        dnn_logit = self.dnn_linear(dnn_output)
        
        # linear & dnn
        if len(self.linear_feature_columns) > 0 and len(self.dnn_feature_columns) > 0:  # linear + dnn
            final_logit = linear_logit + dnn_logit
        # only dnn
        elif len(self.linear_feature_columns) == 0:
            final_logit = dnn_logit
        # only linear
        elif len(self.dnn_feature_columns) == 0:
            final_logit = linear_logit
        else:
            raise NotImplementedError

        y_pred = self.out(final_logit)
        return y_pred
