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
# 实验失败了，这个代码用不上了
#  after_sort中xtrainnew2是排序后的数据，扩增时取了前3000行
#  whiteaug里是9k行扩增后的数据
#  finaltrain是原始数据
#  1. 取出x_train_new2中前3000行数据
#  	1.1 去掉unnamed列
#  2. 将finaltrain中去掉这3000行
#  3. 随机选取finaltrain部分行（60000左右）和whiteaug合并
#  4. 训练

# -------------------------------------
# 1
afterSortWhitePath='/mnt/blockchain0/after_sort/x_train_new2.csv'
afterSortWhite=pd.read_csv(afterSortWhitePath)
# print(afterSortWhite.shape[0])
afterSortWhite=afterSortWhite.drop(columns = 'Unnamed: 0')
afterSortWhite['label']=0
afterSortWhite=afterSortWhite.head(3000)
print(afterSortWhite)
# -------------------------------------
# 2
finalTrainPath='/mnt/blockchain0/zengliang_test/final_train_data/part-00000-41221f5c-bae3-4859-86c5-60e9c9e65f61-c000.csv'
finalTrainData=pd.read_csv(finalTrainPath)
print(finalTrainData.shape[0])
finalTrainData=finalTrainData.loc[finalTrainData['label'] == 0]
print(finalTrainData.shape[0])
# 6364172->63642
finalTrainData=finalTrainData.sample(frac=0.01, random_state=11451)
# 63642->54642
# finalTrainData=finalTrainData.sample(n=54642)
print(finalTrainData.shape[0])
finalTrainData = finalTrainData[~finalTrainData.isin(afterSortWhite.to_dict(orient='list')).all(axis=1)]
print(finalTrainData.shape[0])
# -------------------------------------
# 3
whiteAugPath = '/mnt/blockchain0/after_sort/white_augmentation/x_train_zengqiang.csv'
whiteAugData=pd.read_csv(whiteAugPath)
whiteAugData=whiteAugData.drop(columns = 'Unnamed: 0')
whiteAugData['label']=0

print(whiteAugData)
finalTrainData=finalTrainData.drop(columns = 'address')
finalTrainData=finalTrainData._append(whiteAugData,ignore_index= True)
finalTrainData.to_csv('/mnt/blockchain0/after_sort/finalAugementTrain2.csv',index=False)