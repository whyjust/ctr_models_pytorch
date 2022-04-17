#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   cmpp_test.py
@Time    :   2022/04/17 16:34:58
@Author  :   weiguang 
'''
import os, sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models.ccpm import CCPM
from deepctr_torch.models import *

data = pd.read_csv('./criteo_sample.txt')

sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]

data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0, )
target = ['label']

# 1.Label Encoding for sparse features,and do simple Transformation for dense features
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])

# 2.count #unique features for each sparse field,and record dense feature field name
fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique()) for feat in sparse_features] \
                        + [DenseFeat(feat, 1, ) for feat in dense_features]

dnn_feature_columns = [SparseFeat(feat, data[feat].nunique()) for feat in sparse_features]

feature_names = get_feature_names(fixlen_feature_columns)

# 3.generate input data for model
train, test = train_test_split(data, test_size=0.2, random_state=2020)
train_model_input = {name: train[name] for name in feature_names}
test_model_input = {name: test[name] for name in feature_names}

# 4.Define Model,train,predict and evaluate
device = 'cpu'
use_cuda = True
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = 'cuda:0'

model = CCPM(linear_feature_columns=fixlen_feature_columns, dnn_feature_columns=dnn_feature_columns,
            task='binary', l2_reg_embedding=1e-5, device=device)

model.compile("adagrad", "binary_crossentropy", metrics=["binary_crossentropy", "auc"], )
history = model.fit(train_model_input, train[target].values, batch_size=32, epochs=10, verbose=2,validation_split=0.2)
pred_ans = model.predict(test_model_input, 256)

plt.plot(history.epoch, history.history['loss'])
plt.title('CCPM loss curve')
plt.plot(history.epoch, history.history['val_auc'])
plt.title('CCPM auc curve')
plt.show()

'''
CCPM(
  (embedding_dict): ModuleDict(
    (C1): Embedding(27, 4)
    (C2): Embedding(92, 4)
    (C3): Embedding(172, 4)
    (C4): Embedding(157, 4)
    (C5): Embedding(12, 4)
    (C6): Embedding(7, 4)
    (C7): Embedding(183, 4)
    (C8): Embedding(19, 4)
    (C9): Embedding(2, 4)
    (C10): Embedding(142, 4)
    (C11): Embedding(173, 4)
    (C12): Embedding(170, 4)
    (C13): Embedding(166, 4)
    (C14): Embedding(14, 4)
    (C15): Embedding(170, 4)
    (C16): Embedding(168, 4)
    (C17): Embedding(9, 4)
    (C18): Embedding(127, 4)
    (C19): Embedding(44, 4)
    (C20): Embedding(4, 4)
    (C21): Embedding(169, 4)
    (C22): Embedding(6, 4)
    (C23): Embedding(10, 4)
    (C24): Embedding(125, 4)
    (C25): Embedding(20, 4)
    (C26): Embedding(90, 4)
  )
  (linear_model): Linear(
    (embedding_dict): ModuleDict(
      (C1): Embedding(27, 1)
      (C2): Embedding(92, 1)
      (C3): Embedding(172, 1)
      (C4): Embedding(157, 1)
      (C5): Embedding(12, 1)
      (C6): Embedding(7, 1)
      (C7): Embedding(183, 1)
      (C8): Embedding(19, 1)
      (C9): Embedding(2, 1)
      (C10): Embedding(142, 1)
      (C11): Embedding(173, 1)
      (C12): Embedding(170, 1)
      (C13): Embedding(166, 1)
      (C14): Embedding(14, 1)
      (C15): Embedding(170, 1)
      (C16): Embedding(168, 1)
      (C17): Embedding(9, 1)
      (C18): Embedding(127, 1)
      (C19): Embedding(44, 1)
      (C20): Embedding(4, 1)
      (C21): Embedding(169, 1)
      (C22): Embedding(6, 1)
      (C23): Embedding(10, 1)
      (C24): Embedding(125, 1)
      (C25): Embedding(20, 1)
      (C26): Embedding(90, 1)
    )
  )
  (out): PredictionLayer()
  (conv_layer): ConvLayer(
    (conv_layer): Sequential(
      (0): Conv2dSame(1, 4, kernel_size=(6, 1), stride=(1, 1))
      (1): Tanh()
      (2): KMaxPooling()
      (3): Conv2dSame(4, 4, kernel_size=(5, 1), stride=(1, 1))
      (4): Tanh()
      (5): KMaxPooling()
    )
  )
  (dnn): DNN(
    (dropout): Dropout(p=0, inplace=False)
    (linears): ModuleList(
      (0): Linear(in_features=48, out_features=256, bias=True)
    )
    (activation_layers): ModuleList(
      (0): ReLU(inplace=True)
    )
  )
  (dnn_linear): Linear(in_features=256, out_features=1, bias=False)
)
'''
