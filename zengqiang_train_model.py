# coding=UTF-8
import xgboost as xgb
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
import os
import time
from sklearn import metrics
from sklearn.utils import shuffle
# 对黑样本和白样本都进行增强后对模型进行训练
# 当前是有问题的，还在等白样本增强完成
# train_directoryPath = '/mnt/blockchain0/after_sort/x_train_zengqiang.csv'
# 这个训练的不行，只有1epoch，依托使

# orgin_whitePath= '/mnt/blockchain0/after_sort/x_train_new.csv'
# 此为根据leaf差值排序后生成的12w行数据，要3000行之后的,参杂的未增强数据不能是排序过的
origin_whitePath='/mnt/blockchain0/zengliang_test/final_train_data/part-00000-41221f5c-bae3-4859-86c5-60e9c9e65f61-c000.csv'
train_directoryPath = '/mnt/blockchain0/after_sort/white_augmentation/x_train_zengqiang.csv'
#这个用前3000条增强的9000条，epoch=300

before_trainWhitePath='/mnt/blockchain0/after_sort/x_train_new.csv'
# 取前3000条用来筛除
shaichu=pd.read_csv(before_trainWhitePath)
shaichu=shaichu[0:3000]
shaichu['label']=0.0
shaichu=shaichu.drop(columns = 'Unnamed: 0')
train_df = pd.read_csv(train_directoryPath)
train_df['label'] = 0.0
train_df2=pd.read_csv(origin_whitePath)


train_df2 = train_df2.loc[train_df2['label'] == 0]
train_df2 = train_df2.drop(columns = 'address')
train_df2=train_df2.sample(frac=0.01, random_state=11451)
print(train_df2.shape[0])
# train_df2=train_df2._append()
# train_df2=train_df2.drop_duplicates()
# result_df2 = train_df2[~train_df2.isin(shaichu.to_dict('list')).all(1)]
# duplicates = train_df2.duplicated(subset=shaichu.columns)
# # print(duplicates)
# train_df2 = train_df2[~duplicates]

print(train_df2.shape[0])
train_blackPath="/mnt/blockchain0/after_sort/black_augmentation/black_zengqiang.csv"
train_black=pd.read_csv(train_blackPath)
train_black['label']=1


train_black_x = train_black.drop(columns = 'label')
train_black_y = train_black['label']
# print(type(train_black_y))
train_all = pd.DataFrame()
train_all = train_all._append(train_df,ignore_index= True)
train_all = train_all._append(train_df2,ignore_index= True)
train_all = train_all._append(train_black,ignore_index= True)
train_all=train_all.drop(columns = 'Unnamed: 0')
# train_all = train_all.sample(frac=1, random_state=11451)
# print(train_all)
x_train_new = pd.DataFrame()
y_train_new = pd.DataFrame()
x_train_new= train_all.drop(columns = 'label')
y_train_new = train_all['label']
# 使用索引重置保持索引一一对应



# print(x_train_new)
# print(y_train_new)
# y_train[np.isnan(y_train)]=0
dtrain = xgb.DMatrix(x_train_new,label=y_train_new,enable_categorical=False)

test_directoryPath = '/mnt/blockchain0/zengliang_test/final_test_data/part-00000-9887ecdf-7f72-41d4-8230-bb051ec3a95f-c000.csv'
test_df = pd.read_csv(test_directoryPath)
test_df = test_df.drop(columns = 'address')
x_test= test_df.drop(columns = 'label')
y_test = test_df['label']
# print(x_test)
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
print('-----------------------增强模型训练所用时间为：',time_res)
# model.dump_model("/pub/p1/zengliang_test/model_best.txt")
# model.save_model('/pub/p1/zengliang_test/origin_model_best.json')


# model = xgb.Booster()
# model.load_model('/home/lr/zengliang_test/origin_model_quanliang.json')
#用测试集进行测试
ypred_new = model.predict(dtest)
ypred_new = np.around(ypred_new)
# ypred_new = (ypred_new >= 0.75)*1
print('预测结果中黑样本个数是：',np.sum(ypred_new ==1))
print('新模型的accuracy is ：',metrics.accuracy_score(y_test,ypred_new))
print('新模型的recall is ：',metrics.recall_score(y_test,ypred_new))
print('新模型的precision is ：',metrics.precision_score(y_test,ypred_new))