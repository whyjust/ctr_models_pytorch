import os, sys
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models.afm import AFM
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

# 2.将特征进行标记（sparse或dense）
fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique()) for feat in sparse_features] \
                            + [DenseFeat(feat, 1, ) for feat in dense_features]

# 将sparseFeat特征给dnn，sparseFeat与denseFeat给linear
dnn_feature_columns = [SparseFeat(feat, data[feat].nunique()) for feat in sparse_features] 
linear_feature_columns = fixlen_feature_columns

feature_names = get_feature_names(
    linear_feature_columns + dnn_feature_columns)

# 3.将数据分类
train, test = train_test_split(data, test_size=0.2, random_state=66)
train_model_input = {name: train[name] for name in feature_names}
test_model_input = {name: test[name] for name in feature_names}

device = 'cpu'
use_cuda = True
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = 'cuda:0'

model = AFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
            task='binary',
            l2_reg_embedding=1e-5, device=device)

model.compile("adam", "binary_crossentropy", metrics=["binary_crossentropy", "auc"])

history = model.fit(train_model_input, train[target].values, batch_size=16, epochs=10, verbose=2,validation_split=0.2)
pred_ans = model.predict(test_model_input, 64)

plt.plot(history.epoch, history.history['loss'])
plt.title('afm loss curve')
plt.plot(history.epoch, history.history['val_auc'])
plt.title('afm auc curve')
plt.show()


"""
AFM(
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
  (fm): AFMLayer(
    (dropout): Dropout(p=0, inplace=False)
  )
)
"""
