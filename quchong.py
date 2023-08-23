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
# 此代码专用于去重，然后在从A中排除掉B中已有的行
origin_whitePath='/mnt/blockchain0/zengliang_test/final_train_data/part-00000-41221f5c-bae3-4859-86c5-60e9c9e65f61-c000.csv'
white3000Path='/mnt/blockchain0/after_sort/x_train_new2.csv'

shaichu=pd.read_csv(white3000Path)
shaichu=shaichu[0:3000]
df1=pd.read_csv(origin_whitePath)
print(df1.shape[0])
df1=df1.loc[df1['label'] == 0]
print(df1.shape[0])
# 经测试，取掉address列后会去掉好多行
# df1=df1.drop(columns='address')
df1=df1.drop_duplicates()
print(df1.shape[0])
df1=df1.sample(frac=0.01, random_state=11451)
df1.to_csv('11451.csv')
shaichu.drop(columns = 'Unnamed: 0',axis=1)
# shaichu.drop(columns = 'Unnamed: 0')
shaichu.to_csv('0-3000.csv',index=False)
