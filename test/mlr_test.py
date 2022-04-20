#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   mlr_test.py
@Time    :   2022/04/20 23:56:05
@Author  :   weiguang 
'''
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models.mlr import MLR
from deepctr_torch.models import *

data = pd.read_csv('./criteo_sample.txt')
sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]

data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0, )
target = ['label']

# 1.特征labelencoder与minMaxScaler
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])

# 2.将特征进行分类注册（sparse或dense）
fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique()) for feat in sparse_features] \
                       + [DenseFeat(feat, 1, ) for feat in dense_features]

# deepFM中sparse-embedding向量与dense向量都作为DNN输入
linear_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(fixlen_feature_columns)

# 3.将数据分类
train, test = train_test_split(data, test_size=0.3, random_state=20)
train_model_input = {name: train[name] for name in feature_names}
test_model_input = {name: test[name] for name in feature_names}

device = 'cpu'
use_cuda = True
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = 'cuda:0'

# 4 MLR训练, 其中linear_feature_columns只是为继承基类
model = MLR(fixlen_feature_columns, fixlen_feature_columns, fixlen_feature_columns,
            region_num=4, task='binary', device=device)
model.compile("adam", "binary_crossentropy", metrics=["binary_crossentropy", "auc"])

history = model.fit(train_model_input, train[target].values, batch_size=256, \
                    epochs=10, verbose=2, validation_split=0.2)
pred_ans = model.predict(test_model_input, 256)
print(pred_ans)

plt.plot(history.epoch, history.history['loss'])
plt.title('IFM loss curve')
plt.plot(history.epoch, history.history['val_auc'])
plt.title('IFM auc curve')
plt.show()

