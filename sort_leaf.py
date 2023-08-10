import xgboost as xgb
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
import os
from sklearn import metrics
import time
# 原先用leaf差值对数据进行排序然后和原始数据进行比对精确率的代码，改成了排序后拆选数据的代码

#总训练数据量：6364381,黑样本数据量209条
df_1 = pd.read_csv('/pub/p1/zengliang_test/train_label_1_data.csv')
df_1 = df_1.drop(columns = 'Unnamed: 0')

#整理黑样本数据
df_1_x = df_1.drop(columns = 'address')
df_1_x = df_1_x.drop(columns = 'label')
df_1_y = df_1['label']

x_train_new = pd.DataFrame()
y_train_new = pd.DataFrame()

x_train_new = x_train_new._append(df_1_x,ignore_index= True)
y_train_new = pd.concat([y_train_new,df_1_y],axis = 0)

# #结合样本增强生成的数据
# df_1_add = pd.read_csv('/home/lr/zengliang_test/7003.csv')
# df_1 = pd.concat([df_1,df_1_add],axis =0)

df = pd.read_csv('/pub/p1/zengliang_test/test_1/leaf_result_origin_quanliang_100.csv')
df = df.sort_values(by=['leaf差值'],axis=0,ascending = False)
df = df.drop(columns = 'Unnamed: 0')
df = df.drop(columns = 'leaf差值')

# df = df['第几份']
leaf_list = df.values.tolist()
print(leaf_list[1000][1])


train_df = pd.DataFrame()

#################################################################################
path_0 = '/pub/p1/zengliang_test/group_data/grou_'
path_1 = '.csv'

#根据参数选取一定量的数据，数据量为参数乘100
value = 1200
leaf_list_data = leaf_list[0:value]
for i in range(value): 
    group = leaf_list_data[i][0]
    num = leaf_list_data[i][1]
    group = int(group)
    group = str(group)
    train_filePath = os.path.join(path_0+group+path_1)
    train_df_1 = pd.read_csv(train_filePath,low_memory=False)
    train_df_1.columns = ['address','label','Total_transactions', 'Number_of_received_addresses', 'Number_of_sent_addresses', 'Call_TRC10', 'Min_value_received', 'Max_value_received', 'Avg_value_received', 'Total_ether_sent_for_accounts', 'Min_value_sent', 'Max_value_sent', 'Avg_value_sent', 'Total_ether_received_for_accounts', 'Min_value_sent_to_trc10', 'Max_value_sent_to_trc10', 'Avg_value_sent_to_trc10', 'Total_ether_sent_to_trc10', 'outDegree', 'inDegree', 'Avg_time_between_sent', 'Avg_time_between_received', 'Time_between_first_last', 'mean_Total_transactions', 'mean_Number_of_received_addresses', 'mean_Number_of_sent_addresses', 'mean_Call_TRC10', 'mean_Min_value_received', 'mean_Max_value_received', 'mean_Avg_value_received', 'mean_Total_ether_sent_for_accounts', 'mean_Min_value_sent', 'mean_Max_value_sent', 'mean_Avg_value_sent', 'mean_Total_ether_received_for_accounts', 'mean_Min_value_sent_to_trc10', 'mean_Max_value_sent_to_trc10', 'mean_Avg_value_sent_to_trc10', 'mean_Total_ether_sent_to_trc10', 'mean_outDegree', 'mean_inDegree', 'mean_Avg_time_between_sent', 'mean_Avg_time_between_received', 'mean_Time_between_first_last']
    train_df_1 = train_df_1.loc[num*100:(num+1)*100]
    train_df_new = train_df_1.loc[train_df_1['label']==0]

    x_train_add = train_df_new.drop(columns='label')
    x_train_add = x_train_add.drop(columns = 'address')
    x_train_new = x_train_new._append(x_train_add, ignore_index=True)
    y_train_add = train_df_new['label']
    y_train_new = pd.concat([y_train_new,y_train_add],axis = 0)
    # train_df = pd.concat([train_df,train_df_1],axis = 0)

y_train_new.columns = ['label']
y_train_new.index = range(len(y_train_new))



dtrain_new = xgb.DMatrix(x_train_new,label=y_train_new,enable_categorical=False)

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
watchlist = [(dtrain_new,'train')]

print('-----------------------开始进行训练：-----------------------')
start = time.perf_counter()
model_new = xgb.train(params_0, dtrain_new, num_boost_round=200, evals=watchlist)
end = time.perf_counter()
print('-----------------------训练结束：-----------------------')
time_res = end-start
print('-----------------------新模型训练所用时间为：',time_res)


#准备测试集
test_directoryPath = '/pub/p1/zengliang_test/final_test_data/part-00000-9887ecdf-7f72-41d4-8230-bb051ec3a95f-c000.csv'
test_df = pd.read_csv(test_directoryPath)
print('测试集中样本的数据量是：',len(test_df))
xx = test_df.loc[test_df['label']==1]
print('测试集中黑样本的数据量是：',len(xx))
test_df = test_df.drop(columns='address')
x_test = test_df.drop(columns='label')
y_test = test_df['label']
dtest = xgb.DMatrix(x_test,label=y_test,enable_categorical=False)

#新模型预测结果
ypred_new = model_new.predict(dtest)
print(ypred_new)
ypred_new = (ypred_new >= 0.75)*1
# ypred_new = np.around(ypred_new)

print(ypred_new)
print('预测结果中黑样本个数是：',np.sum(ypred_new ==1))
print('新模型的accuracy is ：',metrics.accuracy_score(y_test,ypred_new))
print('新模型的recall is ：',metrics.recall_score(y_test,ypred_new))
print('原始模型的precision is ：',metrics.precision_score(y_test,ypred_new))
model_new.dump_model("/pub/p1/zengliang_test/model_new_best.txt")
model_new.save_model('/pub/p1/zengliang_test/model_new_best.json')
