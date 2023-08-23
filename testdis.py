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
# 测距离用
modeltxt_path = '/mnt/blockchain0/zengliang_test/model_best.txt'
#获取原模型的leaf值：
f = open(modeltxt_path,'r',encoding='UTF-8')
origin_list =[]
origin_pattern = re.findall(r'(?<=leaf=)([0-9-:\s.]+)$',f.read(),re.M|re.I)
for i in origin_pattern:
    i = float(i)
    origin_list.append(i)
origin_model_leaf = torch.Tensor(origin_list)

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
    print(origin_model_leaf)
    print(new_model_leaf)
    result = dis(origin_model_leaf, new_model_leaf)
    print('------------------------------差值情况是--------------------：',result)
    #运算完删除使用过的txt文档
    # if os.path.exists(newmodeltxt_path):
    #     os.remove(newmodeltxt_path)
    return result

result_now = get_influence(origin_model_leaf, modeltxt_path)
print(result_now)
