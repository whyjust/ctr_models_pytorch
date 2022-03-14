#!/usr/bin/python
# -*- coding:utf-8 -*-
# @FileName  : activation.py
# @Time      : 2022/3/7 15:20
# @Author    : weiguang

import torch
import torch.nn as nn

class Dice(nn.Module):
    """
    DIN中的数据自适应激活函数，可以看作是PReLu的一种推广，可以根据输入数据的分布自适应修正
    输入形状：
        2 dims: [batch_size, embedding_size(features)]
        3 dims: [bathc_size, num_features, embedding_size(features)]
    输出形状:
        2 dims: [batch_size, embedding_size(features)]
        3 dims: [bathc_size, num_features, embedding_size(features)]
    References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
        - https://github.com/zhougr1993/DeepInterestNetwork
        - https://github.com/fanoping/DIN-pytorch
    """
    def __init__(self, emb_size, dim=2, epsilon=1e-8, device='cpu'):
        super(Dice, self).__init__()
        assert dim == 2 or dim == 3, 'dim must be 2 dim or 3 dim'
        # 定义可以将第2维=emb_size进行标准化的函数
        self.bn = nn.BatchNorm1d(emb_size, eps=epsilon)
        # 定义sigmoid激活函数
        self.sigmoid = nn.Sigmoid()
        self.dim = dim

        # 初始化nn.Parameters使得可训练
        if self.dim == 2:
            # 创建一个全为0，形状为(emb_size,)一维数组
            self.alpha = nn.Parameter(torch.zeros((emb_size,)).to(device))
        else:
            # 创建一个全为0，形状为(emb_size,1)二维数组
            self.alpha = nn.Parameter(torch.zeros((emb_size,1)).to(device))

    def forward(self, x):
        assert x.dim() == self.dim
        if self.dim == 2:
            # 将输入x先做batchNorm1d，然后经过sigmoid映射
            # 转换后维度与x一致
            x_p = self.sigmoid(self.bn(x))
            # 相同维度相乘，转换后维度与X维度一致
            out = self.alpha * (1 - x_p) * x + x_p * x
        else:
            # 将x中dim=1与dim=2进行转置，例如(2,3,4)转置为(2,4,3)
            # 目的是为了适配BatchNorm1d, 归一化第二维，因此第二维应为emb_size
            x = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
            # 在将维度恢复至原维度
            out = torch.transpose(out, 1, 2)
        return out

class Identity(nn.Module):
    """
    定义线性激活函数
    """
    def __init__(self, **kwargs):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs

def activation_layer(act_name, hidden_size=None, dice_dim=2):
    """
    定义激活函数layers
    :param act_name: 字符串或nn.Module, 激活函数的名称
    :param hidden_size: 当使用dice时，hidden_size有效
    :param dice_dim: 当使用dice时，hidden_size有效
    :return: 激活函数layer
    """
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'linear':
            act_layer = Identity()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == 'dice':
            act_layer = Dice(hidden_size, dice_dim)
        elif act_name.lower() == 'prelu':
            act_layer = nn.PReLU()
        else:
            raise ValueError('must be sigmoid,linear,relu,dice,prelu!')
    elif isinstance(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError
    return act_layer

if __name__ == '__main__':
    pass
