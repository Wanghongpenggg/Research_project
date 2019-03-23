# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 10:05:23 2019

@author: jie
"""

import pandas as pd
import numpy as np
import ReliefF as rf
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("sd.csv",header=0)
from sklearn.utils import shuffle  
#Data sequence disruption
df = shuffle(df)  
#Get a list of field names
col_names=df.columns.values.tolist() 

x=df[col_names[0:-1]]
print(x.shape)
X = np.array(x)
#Take the last column of the data as the label column
y=df[col_names[-1]]
Y = np.array(y)

r = rf.ReliefF(n_neighbors=100, n_features_to_keep=10)
r_fit = r.fit(X,Y)

narr=np.array(np.absolute(r.feature_scores))
#Construct a sequence with feature importance and feature name
data=pd.Series(narr,index=col_names[0:-1])
print('-----按重要度排序得到前100个特征的重要度----------')
data1=data.sort_values(ascending=False)
print(data1[0:100])
print('-----按重要度排序得到前100个特征------------')
print(data1[0:100].index)

