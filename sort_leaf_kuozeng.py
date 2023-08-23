import xgboost as xgb
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
import os
from sklearn import metrics
import time
# 根据leaf差值对数据进行排序，将排序后的数据保存

#总训练数据量：6364381,黑样本数据量209条
df_1 = pd.read_csv('/mnt/blockchain0/zengliang_test/train_label_1_data.csv')
df_1 = df_1.drop(columns = 'Unnamed: 0')

#整理黑样本数据
df_1_x = df_1.drop(columns = 'address')
df_1_x = df_1_x.drop(columns = 'label')
df_1_y = df_1['label']

x_train_new = pd.DataFrame()
y_train_new = pd.DataFrame()

# x_train_new = x_train_new._append(df_1_x,ignore_index= True)
# y_train_new = pd.concat([y_train_new,df_1_y],axis = 0)

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
path_0 = '/mnt/blockchain0/zengliang_test/group_data/grou_'
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
    # x_train_add = x_train_add.drop(columns = 'address')
    x_train_new = x_train_new._append(x_train_add, ignore_index=True)
    y_train_add = train_df_new['label']
    y_train_new = pd.concat([y_train_new,y_train_add],axis = 0)
    # train_df = pd.concat([train_df,train_df_1],axis = 0)

y_train_new.columns = ['label']
y_train_new.index = range(len(y_train_new))
x_train_new.to_csv('/mnt/blockchain0/after_sort/x_train_new2.csv')
y_train_new.to_csv('/mnt/blockchain0/after_sort/y_train_new2.csv')


