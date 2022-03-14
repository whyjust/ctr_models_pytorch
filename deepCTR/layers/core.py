#!/usr/bin/python
# -*- coding:utf-8 -*-
# @FileName  : core.py
# @Time      : 2022/3/7 16:26
# @Author    : weiguang

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from activation import activation_layer

class LocalActivationUnit(nn.Module):
    """
    该模块为DIN中根据用户兴趣自适应调整不同的候选类目
    输入形状：
        - 2组 3维数组, (batch_size, 1, embedding_size) 和 (batch_size, T, embedding_size)
    输出形状：
        - 3维数组, (batch_size, T, 1)
    References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """
    def __init__(self, hidden_units=(64, 32), embedding_dim=4, activation='sigmoid', dropout_rate=0, dice_dim=3,
                 l2_reg=0, use_bn=False):
        """
        :param hidden_units: attention网络层数及对应的神经元个数，hidden_units=(64, 32)
        :param embedding_dim: embedding维度
        :param activation: 激活函数
        :param dropout_rate: attention网络中随机drop比例
        :param dice_dim: 使用dice有效
        :param l2_reg: 注意力网络中的l2正则项
        :param use_bn: 是否使用batchNorm1d
        """
        super(LocalActivationUnit, self).__init__()
        self.dnn = DNN(inputs_dim=4 * embedding_dim,
                       hidden_units=hidden_units,
                       activation=activation,
                       l2_reg=l2_reg,
                       dropout_rate=dropout_rate,
                       dice_dim=dice_dim,
                       use_bn=use_bn)
        # 输出层, 与最后一层中间层size: (batch, hidden_units[-1])
        self.dense = nn.Linear(hidden_units[-1], 1)

    def forward(self, query, user_behavior):
        # query ad:      size -> batch_size * 1 * embedding_size
        # user behavior: size -> batch_size * time_seq_len * embedding_size
        # size(dim)表示第dim维的长度
        user_behavior_len = user_behavior.size(1)
        # expand只能扩展维度为1的矩阵，也就是说在将query原本为dim=1,expand为user_behavior_len
        # 目的是使queries维度与user_behavior保持一致，方便后续的减与乘运算
        # queries: size -> batch_size * time_seq_len * embedding_size
        queries = query.expand(-1, user_behavior_len, -1)
        # attention_input size: batch_size * time_seq_len * embedding_size*4
        attention_input = torch.cat([queries, user_behavior, queries - user_behavior, \
                                     queries * user_behavior], dim=-1)
        # attention_output为DNN输出size:(batch_size, time_seq_len, hidden_units[-1])
        attention_output = self.dnn(attention_input)
        # 输出层size: (batch_size, time_seq_len, 1)
        attention_score = self.dense(attention_output)
        return attention_score

class DNN(nn.Module):
    """
    多层感知机网络
    输入形状：
        - 多维向量，(batch_size, ..., input_dim), 中间为隐藏层神经元维度,最常见的为二维向量, (batch_size, input_dim)
    输出形状：
        - 多维向量，(batch_size, ...., hidden_size[-1]), 中间为隐藏层，hidden_size[-1]最后一层为输出层，
        - 例如：2维向量输入(batch_size, input_dim)，输出为(batch_size, hidden_size[-1])

    例如：dnn = DNN(inputs_dim=16, hidden_units=(16,64,32))
    Out[9]:
    DNN(
      (dropout): Dropout(p=0, inplace=False)
      (linears): ModuleList(
        (0): Linear(in_features=16, out_features=16, bias=True)
        (1): Linear(in_features=16, out_features=64, bias=True)
        (2): Linear(in_features=64, out_features=32, bias=True)
      )
      (activation_layers): ModuleList(
        (0): ReLU(inplace=True)
        (1): ReLU(inplace=True)
        (2): ReLU(inplace=True)
      )
    )
    """
    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 init_std=0.0001, dice_dim=3, seed=1024, device='cpu'):
        """
        :param inputs_dim: 输入层特征维度
        :param hidden_units: layer个数及对应的神经元个数，hidden_units=(64, 32)
        :param activation: 激活函数
        :param l2_reg: l2正则项
        :param dropout_rate: 随机dropout比例
        :param use_bn: 是否采用BatchNorm1d
        :param init_std: std初始化
        :param dice_dim: dice维度
        :param seed: 随机种子
        :param device: 采用cpu还是gpu计算
        """
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError('hidden_units is empty!')
        hidden_units = [inputs_dim] + list(hidden_units)

        # 通过ModuleList来构建模型，并且注册到网络中
        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i+1]) for i in range(len(hidden_units)-1)]
        )
        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i+1]) for i in range(len(hidden_units)-1)]
            )
        self.activation_layers = nn.ModuleList(
            [activation_layer(activation, hidden_units[i+1], dice_dim) for i in range(len(hidden_units)-1)]
        )

        # ModuleList注册的模型可以访问空间,输出的name为0.weight,0.bias,1.weight,1.bias
        # weight 初始化自定义
        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)
        self.to(device)

    def forward(self, inputs):
        deep_input = inputs
        for i in range(len(self.linears)):
            # 指定对应index的layer层
            fc = self.linears[i](deep_input)
            # 使用对应index的batchNorm1d
            if self.use_bn:
                fc = self.bn[i](fc)
            # 使用对应index的activation_layers激活函数
            fc = self.activation_layers[i](fc)
            # 使用对应index的dropout
            fc = self.dropout(fc)
            # 将中间结果赋值，作为新的输入层
            deep_input = fc
        return deep_input

class PredictionLayer(nn.Module):
    def __init__(self, task='binary', use_bias=True, **kwargs):
        """
        定义layer输出类
        :param task: 任务类型，分类还是回归
        :param use_bias: 是否使用偏置项
        :param kwargs: 其他参数
        """
        if task not in ["binary","multiclass","regression"]:
            raise ValueError("task must be binary,multiclass,regression!")
        super(PredictionLayer, self).__init__()
        self.use_bias = use_bias
        self.task = task
        if self.use_bias:
            # 构建一个偏置项为0的一维数组
            self.bias = nn.Parameter(torch.zeros((1,)))

    def forward(self, X):
        output = X
        if self.use_bias:
            output += self.bias
        if task == 'binary':
            output = torch.sigmoid(output)
        return output

class Conv2dSame(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        """
        Con2d上层Same封装器
        :param in_channels: 输入通道数取决与输入数据，例如RGB图片，in_channel固定为3
        :param out_channels: 过滤器的数量也就是从不同的卷积核的个数， 4个卷积核则输出通道为4维
        :param kernel_size: 卷积核的尺寸
        :param stride: 步长
        :param padding: padding补数据
        :param dilation:
        :param groups:
        :param bias:
        """
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation,
            groups, bias)
        # 将self.weight均匀分布
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # 获取height与with
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        # self.stride=(3,2)代表竖直步长为3，水平步长为2
        oh = math.ceil(ih / self.stride[0])
        ow = math.ceil(iw / self.stride[1])
        # 确定需要是否需要补数
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h>0 or pad_w>0:
            # 执行pad补数操作
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        out = F.conv2d(x, self.weight, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
        return out


if __name__ == '__main__':
    pass
    # embedding_dim = 32
    # batch_size = 64
    # time_seq_len = 20
    # query = torch.randn((batch_size, 1, embedding_dim))     # (64, 1, 32)
    # user_behavior = torch.randn((batch_size, time_seq_len, embedding_dim))      # (64, 20, 32)
    # local_activation_unit = LocalActivationUnit(embedding_dim=embedding_dim)
    # print(local_activation_unit(query, user_behavior).shape)        # (64, 20, 1)

    # con2d
    # Conv2dSame(in_channels=3, out_channels=2, kernel_size=(3, 1), stride=1)
