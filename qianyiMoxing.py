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

finalWhitePath='/mnt/blockchain0/after_sort/finalAugementTrain3.csv'
train_white=pd.read_csv(finalWhitePath)







# 在去重后再删除address列

train_blackPath="/mnt/blockchain0/after_sort/black_augmentation/black_zengqiang.csv"
train_black=pd.read_csv(train_blackPath)
train_black['label']=1
train_black=train_black.drop(columns = 'Unnamed: 0')


train_all = pd.DataFrame()
train_all = train_all._append(train_white,ignore_index= True)
train_all = train_all._append(train_black,ignore_index= True)

x_train_new = pd.DataFrame()
y_train_new = pd.DataFrame()
x_train_new= train_all.drop(columns = 'label')
drop_columns=['Call_TRC10','Min_value_sent_to_trc10','Max_value_sent_to_trc10','Avg_value_sent_to_trc10','Total_ether_sent_to_trc10','outDegree','inDegree','mean_Call_TRC10','mean_Min_value_sent_to_trc10','mean_Max_value_sent_to_trc10','mean_Avg_value_sent_to_trc10','mean_Total_ether_sent_to_trc10','mean_outDegree','mean_inDegree']
# x_train_new=x_train_new.drop(columns=drop_columns)
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
# x_test=x_test.drop(columns=drop_columns)
y_test = test_df['label']
# print(x_test)
dtest = xgb.DMatrix(x_test,label=y_test,enable_categorical=False)

params_0={'booster':'gbtree',
    'objective': 'multi:softmax',
    'num_class':'3',
    # 'objective':'binary:logistic',
    'eval_metric': 'auc',
    'max_depth':3,
    'lambda':5,
    'alpha':0,
    'subsample':1,
    'colsample_bytree':0.8,
    'colsample_bylevel':0.6,
    'min_child_weight':2,
    'eta': 0.1,
    'seed':0,
    'nthread':16,
     'silent':1,
     'scale_pos_weight':10
    # 'process_type': 'update',
    # 'updater': 'refresh',
    # 'refresh_leaf': True
    
}
# params_0={'booster':'gbtree',
#     'objective': 'multi:softmax',
#     'num_class':'3',
#     # 'objective':'binary:logistic',
#     'eval_metric': 'auc',
#     'max_depth':4,
#     'lambda':1,
#     'alpha':1,
#     'subsample':0.8,
#     'colsample_bytree':0.8,
#     'colsample_bylevel':0.6,
#     'min_child_weight':2,
#     'eta': 0.1,
#     'seed':0,
#     'nthread':16,
#      'silent':1
#     # 'process_type': 'update',
#     # 'updater': 'refresh',
#     # 'refresh_leaf': True
    
# }
watchlist = [(dtrain,'train')]
boostArray=np.array([300,400,500,600,1000])
# boostArray=np.array([10,50,10,150,200,250])
# boostArray=np.array([350,360,370,380,390])
# boostArray=np.array([1100,1200,1500,1600,2000])
for i in range(5):
  print('-----------------------开始进行训练：-----------------------')
  start = time.perf_counter()
  model = xgb.train(params_0, dtrain, num_boost_round=boostArray[i], evals=watchlist)
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
  # print(ypred_new)
  # ypred_new = (ypred_new >= 0.75)*1
  blackCnt=np.sum(ypred_new ==1)
  accuracy=metrics.accuracy_score(y_test,ypred_new)
  recall=metrics.recall_score(y_test,ypred_new,  labels=[1], average='micro')
  precision=metrics.precision_score(y_test,ypred_new,  labels=[1], average='micro')
  f1Score=metrics.f1_score(y_test,ypred_new, labels=[1], average='micro')
  print('预测结果中黑样本个数是：',blackCnt)
  print('新模型的accuracy is ：',accuracy)
  print('新模型的recall is ：',recall)
  print('新模型的precision is ：',precision)
  print('新模型的f1_score is ：',f1Score)
  # print(' ')
  file_path = "logistic.txt"

  # 打开文件并写入F1-Score值
  with open(file_path, "a") as file:
    file.write(f"num_boost_round: {boostArray[i]}\n")
    file.write(f"blackCnt: {blackCnt}\n")
    file.write(f"accuracy: {accuracy}\n")
    file.write(f"recall: {recall}\n")
    file.write(f"precision: {precision}\n")
    file.write(f"f1_score: {f1Score}\n")
    file.write("\n")
  # boostArray[i]=metrics.precision_score(y_test,ypred_new,  labels=[1], average='micro')
  # print('新模型的precision is ：',metrics.precision_score(y_test,ypred_new))