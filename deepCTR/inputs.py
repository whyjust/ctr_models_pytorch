#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   inputs.py
@Time    :   2022/03/16 23:29:43
@Author  :   weiguang 
'''
from collections import OrderedDict, namedtuple, defaultdict
from itertools import chain
from pip import main
import torch
import torch.nn as nn
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from layers.sequence import SequencePoolingLayer
from layers.utils import concat_fun

DEFAULT_GROUP_NAME = "default_group"
class SparseFeat(namedtuple('SparseFeat',
                ['name', 'vocabulary_size', 'embedding_dim', \
                 'use_hash', 'dtype', 'embedding_name', 'group_name'])):
    """
    稀疏特征类注册
    输入: 原始特征名称及相关的属性值
    输出: SparseFeat类型的特征
    """
    __slots__ = ()
    
    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, \
        dtype='int32', embedding_name=None, group_name=DEFAULT_GROUP_NAME):
        # embedding_name如果为None,则将name赋值给embedding_name
        if embedding_name is None:
            embedding_name = name
        # 特征维度
        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        
        if use_hash:
            print("Notice! Feature Hashing on the fly currently is not supported in torch version,you can use tensorflow version!")
        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, \
            use_hash, dtype, embedding_name, group_name)
    
    def __hash__(self):
        return self.name.__hash__()
    
class VarLenSparseFeat(namedtuple('VarLenSparseFeat', ['sparsefeat', 'maxlen', 'combiner', 'length_name'])):
    """
    结合sparseFeat, max_len, combiner, length_name注册为新类VarLenSparseFeat
    输入: sparseFeat及相关的max_len/combiner/length_name
    输出: VarLenSparseFeat类型注册
    """
    __slots__ = ()

    def __new__(cls, sparsefeat, maxlen, combiner='mean', length_name=None):
        return super(VarLenSparseFeat, cls).__new__(cls, sparsefeat, maxlen, combiner, length_name)

    @property
    def name(self):
        return self.sparsefeat.name
    
    @property
    def vocabulary_size(self):
        return self.sparsefeat.vocabulary_size
    
    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_size
    
    @property
    def use_hash(self):
        return self.sparsefeat.use_hash
    
    @property
    def dtype(self):
        return self.sparsefeat.dtype
    
    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name
    
    @property
    def group_name(self):
        return self.sparsefeat.group_name
    
    def __hash__(self):
        return self.name.__hash__()
    
class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype'])):
    """
    数值型特征类
    输入: 特征及对应的属性名
    输出: DenseFeat类
    """
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype='float32'):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)
    
    def __hash__(self):
        return self.name.__hash__()

# 获取feature名称
def get_feature_names(feature_columns):
    features = build_input_features(feature_columns)
    return list(features.keys())

# def get_inputs_list(inputs):
#     return list(chain(*list(map(lambda x: x.values(), filter(lambda x: x is not None, inputs)))))

def build_input_features(feature_columns):
    """
    构建输出输入特征
    输入:
        - 输入特征列表
    输出:
        - 输入 {feature_name: (start, start+dimension)}
    """
    features = OrderedDict()
    start = 0
    # 特征列表
    for feat in feature_columns:
        feat_name = feat.name
        if feat_name in features:
            continue
        # 如果feat类型为SparseFeat
        if isinstance(feat, SparseFeat):
            features[feat_name] = (start, start + 1)
            start += 1
        # 如果feat类型为DenseFeat
        elif isinstance(feat, DenseFeat):
            features[feat_name] = (start, start + feat.dimension)
            start += feat.dimension
        # 如果feat类型为VarLenSparseFeat
        elif isinstance(feat, VarLenSparseFeat):
            features[feat_name] = (start, start + feat.maxlen)
            start += feat.maxlen
            if feat.length_name is not None and feat.length_name not in features:
                features[feat.length_name] = (start, start + 1)
                start += 1
        else:
            raise TypeError("Invalid feature column type,got", type(feat))
    return features


def combined_dnn_input(sparse_embedding_list, dense_value_list):
    """
    组合dnn特征
    Args:
        sparse_embedding_list (_type_): sparse_embedding列表
        dense_value_list (_type_): dense稠密列表

    Raises:
        NotImplementedError: 执行错误

    Returns:
        _type_: torch
    """
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        # 将sparse_embedding_list按照dim=-1合并
        # flattern将dim=1后的维度拉平
        sparse_dnn_input = torch.flatten(
            torch.cat(sparse_embedding_list, dim=-1), start_dim=1
        )
        dense_dnn_input = torch.flatten(
            torch.cat(dense_value_list, dim=-1), start_dim=1
        )
        # concat_fun按照dim的维度合并
        return concat_fun([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return torch.flatten(
            torch.cat(sparse_embedding_list, dim=-1), start_dim=1
        )
    elif len(dense_value_list) > 0:
        return torch.flatten(
            torch.cat(dense_value_list, dim=-1), start_dim=1
        )
    else:
        raise NotImplementedError


def get_varlen_pooling_list(embedding_dict, features, feature_index, varlen_sparse_feature_columns, device):
    """
    获取特征pooling列表
    Args:
        embedding_dict (_type_): embedding字典
        features (_type_): 特征
        feature_index (_type_): 特征index
        varlen_sparse_feature_columns (_type_): varSparse特征
        device (_type_): 设备

    Returns:
        _type_: _description_
    """
    varlen_sparse_embedding_list = []
    for feat in varlen_sparse_feature_columns:
        seq_emb = embedding_dict[feat.name]
        if feat.length_name is None:
            seq_mask = features[:, feature_index[feat.name][0]:feature_index[feat.name][1]].long() != 0
            # emb size -> (batch_size, 1, embedding_size)
            # 如果feat的length_name为None, support_masking生效
            emb = SequencePoolingLayer(mode=feat.combiner, supports_masking=seq_mask, device=device)(
                [seq_emb, seq_mask]
            )
        else:
            seq_length = features[:, feature_index[feat.length_name][0]:feature_index[feat.length_name][1]].long()
            # emb size -> (batch_size, 1, embedding_size)
            # 如果feat的length_name为None, support_masking为False
            emb = SequencePoolingLayer(model=feat.combiner, supports_masking=False, device=device)(
                [seq_emb, seq_length]
            )
        varlen_sparse_embedding_list.append(emb)
    return varlen_sparse_embedding_list


def create_embedding_matrix(feature_columns, init_std=0.0001, linear=False, sparse=False, device='cpu'):
    """
    Args:
        feature_columns (_type_): 特征列
        init_std (float, optional): 初始化值. Defaults to 0.0001.
        linear (bool, optional): 线性. Defaults to False.
        sparse (bool, optional): 稀疏. Defaults to False.
        device (str, optional): 设备. Defaults to 'cpu'.
    """
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)
    ) if len(feature_columns) else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)
    ) if len(feature_columns) else []
    embedding_dict = nn.ModuleDict(
        {feat.embedding_name: nn.Embedding(feat.vocabulary_size, feat.embedding_size if not linear else 1, sparse=sparse)
        for feat in sparse_feature_columns + varlen_sparse_feature_columns}
    )
    
    # for feat in varlen_sparse_feature_columns:
    #     embedding_dict[feat.embedding_name] = nn.EmbeddingBag(
    #         feat.dimension, embedding_size, sparse=sparse, mode=feat.combiner)

    for tensor in embedding_dict.values():
        nn.init.normal_(tensor.weight, mean=0, std=init_std)
    
    # return nn.ModuleDict: for sparse features, {embedding_name: nn.Embedding}
    # for varlen sparse features, {embedding_name: nn.EmbeddingBag}
    return embedding_dict.to(device)

def embedding_lookup(X, sparse_embedding_dict, sparse_input_dict, sparse_feature_columns, return_feat_list=(), to_list=False):
    """
    embedding lookup查询表
    Args:
        X (_type_): 输入形状, (batch_size, hidden_dim)
        sparse_embedding_dict (_type_): nn.ModuleDict, {embedding_name: nn.Embedding}
        sparse_input_dict (_type_): OrderedDict, {feature_name:(start, start+dimension)}}
        sparse_feature_columns (_type_): list, 稀疏特征名
        to_list (bool, optional): list, 在hash转换中被masked的特征名. Defaults to False.
    Return:
        group_embedding_dict: defaultdict(list)
    """
    group_embedding_dict = defaultdict(list)
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if (len(return_feat_list) == 0 or feature_name in return_feat_list):
            # TODO: add hash function
            # 获取index
            lookup_idx = np.array(sparse_input_dict[feature_name])
            # 根据index取对应input tensor
            input_tensor = X[:, lookup_idx[0]:lookup_idx[1]].long()
            # 获取对应nn.embedding
            emb = sparse_embedding_dict[embedding_name](input_tensor)
            group_embedding_dict[fc.group_name].append(emb)
    if to_list:
        # 将多个列表通过chain连接一起
        return list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict

def varlen_embedding_lookup(X, embedding_dict, sequence_input_dict, varlen_sparse_feature_coulmns):
    """
    varlen的embedding查找lookup
    Args:
        X (_type_): 特征
        embedding_dict (_type_): embedding字典
        sequence_input_dict (_type_): 输入序列字典
        varlen_sparse_feature_coulmns (_type_): varlen稀疏特征

    Returns:
        _type_: 字典
    """
    varlen_embedding_vec_dict = {}
    for fc in varlen_sparse_feature_coulmns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.use_hash:
            # lookup_idx = Hash(fc.vocabulary_size, mask_zero=True)(sequence_input_dict[feature_name])
            # TODO: add hash function
            lookup_idx = sequence_input_dict[feature_name]
        else:
            lookup_idx = sequence_input_dict[feature_name]
        # 根据lookup查找embedding
        varlen_embedding_vec_dict[feature_name] = embedding_dict[embedding_name](
            X[:, lookup_idx[0]:lookup_idx[1]].long()
        )
    return varlen_embedding_vec_dict

def get_dense_input(X, features, feature_columns):
    """
    获取dense输入
    Args:
        X (_type_): _description_
        features (_type_): _description_
        feature_columns (_type_): _description_

    Returns:
        _type_: dense输入列表
    """
    dense_feature_columns = list(filter(
        lambda x: isinstance(x, DenseFeat), feature_columns
    )) if feature_columns else []
    dense_input_list = []
    for fc in dense_feature_columns:
        lookup_idx = np.array(features[fc.name])
        input_tensor = X[:, lookup_idx[0]:lookup_idx[1]].float()
        dense_input_list.append(input_tensor)
    return dense_input_list


def maxlen_lookup(X, sparse_input_dict, maxlen_column):
    """
    maxlen lookup查找表
    Args:
        X (_type_): 特征
        sparse_input_dict (_type_): sparse输入dict
        maxlen_column (_type_): maxlen_column

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if maxlen_column is None or len(maxlen_column)==0:
        raise ValueError('please add max length column for VarLenSparseFeat of DIN/DIEN input')
    lookup_idx = np.array(sparse_input_dict[maxlen_column[0]])
    return X[:, lookup_idx[0]:lookup_idx[1]].long()


if __name__ == "__main__":
    sparse_feature_names = ['C1', 'C2', 'C3', 'C4']
    dense_feature_names = ['D1', 'D2', 'D3', 'D4']
    sparse_columns = [SparseFeat(feat, 10, 32) for feat in sparse_feature_names]
    dense_columns = [DenseFeat(feat, 1) for feat in dense_feature_names]
    var_len_sparse_columns = [VarLenSparseFeat(sp_c, 20, 'mean') for sp_c in sparse_columns]
    print("sparse columns:", sparse_columns)
    print("dense columns:", dense_columns)
    print("var_len_sparse feats:", var_len_sparse_columns)

    mix_feature_columns = sparse_columns + dense_columns
    # 建立features
    mix_feature = build_input_features(mix_feature_columns)
    print("mix_feature:", mix_feature)

    # 新建sparseFeat
    sparseV2columns = [SparseFeat(feat, 20, 10) for feat in ['F1', 'F2', 'F3']]
    embedding_dict = create_embedding_matrix(sparseV2columns+var_len_sparse_columns, init_std=0.0001, linear=True, sparse=True)
    print("embedding dict:", embedding_dict)

    # 

