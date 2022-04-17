#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   din_test.py
@Time    :   2022/04/17 21:15:11
@Author  :   weiguang 
'''
import os, sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names
from deepctr_torch.models.din import DIN
from deepctr_torch.models import *
#--------------
## 构造数据
#--------------
hash_flag = False
feature_columns = [SparseFeat('user', 4, embedding_dim=4, use_hash=hash_flag),
                   SparseFeat('gender', 2, embedding_dim=4, use_hash=hash_flag),
                   SparseFeat('item_id', 3 + 1, embedding_dim=8, use_hash=hash_flag),
                   SparseFeat('cate_id', 2 + 1, embedding_dim=4, use_hash=hash_flag),
                   DenseFeat('pay_score', 1)]

feature_columns += [
    VarLenSparseFeat(SparseFeat('hist_item_id', vocabulary_size=3 + 1, embedding_dim=8, embedding_name='item_id'),
                        maxlen=4, length_name="seq_length"),
    VarLenSparseFeat(SparseFeat('hist_cate_id', vocabulary_size=2 + 1, embedding_dim=4, embedding_name='cate_id'),
                        maxlen=4, length_name="seq_length")]

behavior_feature_list = ["item_id", "cate_id"]
uid = np.array([0, 1, 2, 3])
gender = np.array([0, 1, 0, 1])
item_id = np.array([1, 2, 3, 2])  # 0 is mask value
cate_id = np.array([1, 2, 1, 2])  # 0 is mask value
score = np.array([0.1, 0.2, 0.3, 0.2])

# 维度为4x4
hist_item_id = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0], [1, 2, 0, 0]])
hist_cate_id = np.array([[1, 1, 2, 0], [2, 1, 1, 0], [2, 1, 0, 0], [1, 2, 0, 0]])
behavior_length = np.array([3, 3, 2, 2])

# 数据装载为dict，因此存在二维数据，所以不用DataFrame
feature_dict = {'user': uid, 'gender': gender, 'item_id': item_id, 'cate_id': cate_id,
                'hist_item_id': hist_item_id, 'hist_cate_id': hist_cate_id,
                'pay_score': score, "seq_length": behavior_length}

x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
y = np.array([1, 0, 1, 0])
model_name = "DIN"
model = DIN(feature_columns, behavior_feature_list, dnn_dropout=0.5, device='cpu')

model.compile('adam', 'binary_crossentropy', metrics=['binary_crossentropy'])
model.fit(x, y, batch_size=100, epochs=1, validation_split=0.5)

print(model_name + 'test, train valid pass!')
torch.save(model.state_dict(), model_name + '_weights.h5')
model.load_state_dict(torch.load(model_name + '_weights.h5'))
os.remove(model_name + '_weights.h5')
print(model_name + 'test save load weight pass!')

check_model_io = True
if check_model_io:
    torch.save(model, model_name + '.h5')
    model = torch.load(model_name + '.h5')
    os.remove(model_name + '.h5')
    print(model_name + 'test save load model pass!')
print(model_name + 'test pass!')

'''
DIN(
  (embedding_dict): ModuleDict(
    (user): Embedding(4, 4)
    (gender): Embedding(2, 4)
    (item_id): Embedding(4, 8)
    (cate_id): Embedding(3, 4)
  )
  (linear_model): Linear(
    (embedding_dict): ModuleDict()
  )
  (out): PredictionLayer()
  (attention): AttentionSequencePoolingLayer(
    (local_att): LocalActivationUnit(
      (dnn): DNN(
        (dropout): Dropout(p=0, inplace=False)
        (linears): ModuleList(
          (0): Linear(in_features=48, out_features=64, bias=True)
          (1): Linear(in_features=64, out_features=16, bias=True)
        )
        (activation_layers): ModuleList(
          (0): Dice(
            (bn): BatchNorm1d(64, eps=1e-08, momentum=0.1, affine=True, track_running_stats=True)
            (sigmoid): Sigmoid()
          )
          (1): Dice(
            (bn): BatchNorm1d(16, eps=1e-08, momentum=0.1, affine=True, track_running_stats=True)
            (sigmoid): Sigmoid()
          )
        )
      )
      (dense): Linear(in_features=16, out_features=1, bias=True)
    )
  )
  (dnn): DNN(
    (dropout): Dropout(p=0.5, inplace=False)
    (linears): ModuleList(
      (0): Linear(in_features=33, out_features=256, bias=True)
      (1): Linear(in_features=256, out_features=128, bias=True)
    )
    (activation_layers): ModuleList(
      (0): ReLU(inplace=True)
      (1): ReLU(inplace=True)
    )
  )
  (dnn_linear): Linear(in_features=128, out_features=1, bias=False)
)
'''
