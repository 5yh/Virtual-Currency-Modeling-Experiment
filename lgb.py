import lightgbm as lgb
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
import os
import time
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score,fbeta_score
#作用未知


train_directoryPath = '/pub/p1/zengliang_test/final_train_data/part-00000-41221f5c-bae3-4859-86c5-60e9c9e65f61-c000.csv'
train_df = pd.read_csv(train_directoryPath)
train_df_1 = train_df.loc[train_df['label'] == 1]
train_df_0 = train_df.loc[train_df['label'] == 0]
train_df_0 = train_df_0.sample(frac = 0.01)
train_df = train_df_0._append(train_df_1)
train_df = train_df.drop(columns = 'address')
x_train = train_df.drop(columns = 'label')
y_train = train_df['label']
# y_train[np.isnan(y_train)]=0
lgb_train = lgb.Dataset(x_train,y_train)

test_directoryPath = '/pub/p1/zengliang_test/final_test_data/part-00000-9887ecdf-7f72-41d4-8230-bb051ec3a95f-c000.csv'
test_df = pd.read_csv(test_directoryPath)
test_df = test_df.drop(columns = 'address')
x_test= test_df.drop(columns = 'label')
y_test = test_df['label']
lgb_eval = lgb.Dataset(x_test, y_test, reference = lgb_train)

params = {
            'boosting_type': 'gbdt',
            #'boosting_type': 'dart',
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'min_child_weight': 1.5,
            'max_depth':3,
            'min_data_in_leaf':5000,
            'num_leaves': 2**2,
            'lambda_l2': 10,
            'subsample': 1.0,
            'colsample_bytree': 0.7,
            'colsample_bylevel': 0.7,
            'learning_rate': 0.001,
            'tree_method': 'exact',
            'seed': 2023,
            "num_class": 2,
            'silent': True,
            'verbose': -1,
}
#训练参数设置
gbm = lgb.train(params,lgb_train,num_boost_round=5000)

lgb_pre = gbm.predict(x_test)
print(lgb_pre)
lgb_pre = lgb_pre[:,1] 
y_pred = (lgb_pre >= 0.5)*1
# y_pred = np.around(lgb_pre).astype(int)
print(y_pred)
print('预测结果中黑样本个数是：',np.sum(y_pred ==1))
accuracy = accuracy_score(y_test,y_pred)
print('accuracy is :',accuracy)
recall = recall_score(y_test,y_pred)
print('recall is :',recall)
precision = precision_score(y_test,y_pred)
print('precision is :',precision)
f1_score = 2*(precision*recall)/(precision+recall)
print('f1_score is :',f1_score)
