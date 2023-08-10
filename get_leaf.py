import os.path
import glob
import xgboost as xgb
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re 
import os
import time




params_1={'booster':'gbtree',
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
    'process_type': 'update',
    'updater': 'refresh',
    'refresh_leaf': True
}


model_path = "/pub/p1/zengliang_test/origin_model_best.json"
modeltxt_path = '/pub/p1/zengliang_test/model_best.txt'
newmodeltxt_path = '/pub/p1/zengliang_test/newmodel.txt'
#获取原模型的leaf值：
f = open(modeltxt_path,'r',encoding='UTF-8')
origin_list =[]
origin_pattern = re.findall(r'(?<=leaf=)([0-9-:\s.]+)$',f.read(),re.M|re.I)
for i in origin_pattern:
    i = float(i)
    origin_list.append(i)
origin_model_leaf = torch.Tensor(origin_list)

#计算leaf差值函数，用欧式距离比较差值情况
def get_influence(origin_model_leaf,newmodeltxt_path):
    f_new = open(newmodeltxt_path,'r',encoding='UTF-8')
    new_list =[]
    new_pattern = re.findall(r'(?<=leaf=)([0-9-:\s.]+)$',f_new.read(),re.M|re.I)
    for k in new_pattern:
        k = float(k)
        new_list.append(k)
    new_model_leaf = torch.Tensor(new_list)
    # print('new_leaf is :',new_model_leaf.shape)
    # print('origin leaf is :',origin_model_leaf.shape)
    dis = nn.PairwiseDistance(p=2)
    result = dis(origin_model_leaf, new_model_leaf)
    print('------------------------------差值情况是--------------------：',result)
    #运算完删除使用过的txt文档
    if os.path.exists(newmodeltxt_path):
        os.remove(newmodeltxt_path)
    return result

def re_train(model_path,single_data,num_round=200):
    model = xgb.Booster()
    model.load_model(model_path)
    # model.dump_model("model.txt")
    print("retrain中")
    new_model = xgb.train(params_1, single_data, num_round, evals=[(single_data,'train')], xgb_model=model_path)#原有模型基础上继续训练
    new_model.dump_model("/pub/p1/zengliang_test/newmodel.txt")
    

result_df = pd.DataFrame()

for m in range(10):
    path_0 = '/pub/p1/zengliang_test/group_data/grou_'
    path_1 = '.csv'
    m =str(m)
    train_filePath = os.path.join(path_0+m+path_1)
    m = int(m)
    # train_filePath =  '/home/lr/zengliang_test/group_data/grou-*.csv'
    train_df = pd.read_csv(train_filePath,low_memory=False)
    train_df.columns = ['address','label','Total_transactions', 'Number_of_received_addresses', 'Number_of_sent_addresses', 'Call_TRC10', 'Min_value_received', 'Max_value_received', 'Avg_value_received', 'Total_ether_sent_for_accounts', 'Min_value_sent', 'Max_value_sent', 'Avg_value_sent', 'Total_ether_received_for_accounts', 'Min_value_sent_to_trc10', 'Max_value_sent_to_trc10', 'Avg_value_sent_to_trc10', 'Total_ether_sent_to_trc10', 'outDegree', 'inDegree', 'Avg_time_between_sent', 'Avg_time_between_received', 'Time_between_first_last', 'mean_Total_transactions', 'mean_Number_of_received_addresses', 'mean_Number_of_sent_addresses', 'mean_Call_TRC10', 'mean_Min_value_received', 'mean_Max_value_received', 'mean_Avg_value_received', 'mean_Total_ether_sent_for_accounts', 'mean_Min_value_sent', 'mean_Max_value_sent', 'mean_Avg_value_sent', 'mean_Total_ether_received_for_accounts', 'mean_Min_value_sent_to_trc10', 'mean_Max_value_sent_to_trc10', 'mean_Avg_value_sent_to_trc10', 'mean_Total_ether_sent_to_trc10', 'mean_outDegree', 'mean_inDegree', 'mean_Avg_time_between_sent', 'mean_Avg_time_between_received', 'mean_Time_between_first_last']
    train_df = train_df.drop(columns='address')
    print('数据量是：',len(train_df))
    x_train = train_df.drop(columns='label')
    y_train = train_df['label']
    # counts_zero = y_train[y_train==0.0].shape[0]
    # print("几个0.0：",counts_zero)
    y_train_not_zero=y_train[y_train!=0.0]
    print(y_train_not_zero)
    # train_df = train_df.iloc[:,1:]
    # x_train = train_df.iloc[:,0:42]
    # y_train = train_df.iloc[:,-1]
    for i in range(train_df.shape[0]//100):
        x_train_re = x_train[i*100:(i+1)*100]
        y_train_re = y_train[i*100:(i+1)*100]
        # print(x_train_re)
        # print(y_train_re)
        single_data = xgb.DMatrix(x_train_re, label=y_train_re, enable_categorical=True)
    #再训练
        re_train(model_path, single_data)
        print("你是")
        result_now = get_influence(origin_model_leaf, newmodeltxt_path)
        print('训练数据行数是：',len(x_train_re))
        print('数据量是：',len(train_df))
        result_dict = {'簇':m,'第几份':i,'leaf差值':result_now}
        result_df = result_df.append(result_dict,ignore_index=True)

result_df.to_csv('/pub/p1/zengliang_test/test_1/leaf_result_origin_best.csv')
