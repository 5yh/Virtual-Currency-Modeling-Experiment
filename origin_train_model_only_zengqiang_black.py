# coding=UTF-8
# 仅增强黑样本
import xgboost as xgb
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
import os
import time
from sklearn import metrics


train_directoryPath = '/mnt/blockchain0/zengliang_test/final_train_data/part-00000-41221f5c-bae3-4859-86c5-60e9c9e65f61-c000.csv'
train_df = pd.read_csv(train_directoryPath)
# train_df_1 = train_df.loc[train_df['label'] == 1]
train_df_0 = train_df.loc[train_df['label'] == 0]
train_blackPath="/mnt/blockchain0/after_sort/black_augmentation/black_zengqiang.csv"
train_black=pd.read_csv(train_blackPath)
train_black['label']=1
train_black=train_black.drop(columns = 'Unnamed: 0')
# train_df_1 = train_black.loc[train_black['label'] == 1]
train_df_1=train_black
train_df_0 = train_df_0.sample(frac = 0.01, random_state=11451)
print(train_df_0)
train_df = train_df_0._append(train_df_1)
# train_df = train_df.sample(frac=1, random_state=114514)
train_df = train_df.drop(columns = 'address')
# print(train_df)
x_train = train_df.drop(columns = 'label')
y_train = train_df['label']
print(y_train)
# y_train[np.isnan(y_train)]=0
dtrain = xgb.DMatrix(x_train,label=y_train,enable_categorical=False)

test_directoryPath = '/mnt/blockchain0/zengliang_test/final_test_data/part-00000-9887ecdf-7f72-41d4-8230-bb051ec3a95f-c000.csv'
test_df = pd.read_csv(test_directoryPath)
test_df = test_df.drop(columns = 'address')
x_test= test_df.drop(columns = 'label')
y_test = test_df['label']
dtest = xgb.DMatrix(x_test,label=y_test,enable_categorical=False)

params_0={'booster':'gbtree',
    # 'objective': 'multi:softmax',
    # 'num_class':'2',
    'objective':'binary:logistic',
    'eval_metric': 'auc',
    'max_depth':3,
    'lambda':10,
    'subsample':1,
    'colsample_bytree':0.8,
    'colsample_bylevel':0.6,
    'min_child_weight':2,
    'eta': 0.1,
    'seed':0,
    'nthread':8,
     'silent':1,
    # 'process_type': 'update',
    # 'updater': 'refresh',
    # 'refresh_leaf': True
}
watchlist = [(dtrain,'train')]

print('-----------------------开始进行训练：-----------------------')
start = time.perf_counter()
model = xgb.train(params_0, dtrain, num_boost_round=200, evals=watchlist)
end = time.perf_counter()
print('-----------------------训练结束：-----------------------')
time_res = end-start
print('-----------------------原始模型训练所用时间为：',time_res)
# model.dump_model("/pub/p1/zengliang_test/model_best.txt")
# model.save_model('/pub/p1/zengliang_test/origin_model_best.json')


# model = xgb.Booster()
# model.load_model('/home/lr/zengliang_test/origin_model_quanliang.json')
#用测试集进行测试
ypred_new = model.predict(dtest)
ypred_new = np.around(ypred_new)
# ypred_new = (ypred_new >= 0.75)*1
print('预测结果中黑样本个数是：',np.sum(ypred_new ==1))
print('只增强黑模型的accuracy is ：',metrics.accuracy_score(y_test,ypred_new))
print('只增强黑模型的recall is ：',metrics.recall_score(y_test,ypred_new))
print('只增强黑模型的precision is ：',metrics.precision_score(y_test,ypred_new))