# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 20:49:56 2019

@author: jie
"""

import pandas as pd
import numpy as np
import csv as csv
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


#利用LassoCV进行特征选择
df = pd.read_csv("sd.csv",header=0)
from sklearn.utils import shuffle  
#数据顺序打乱
df = shuffle(df)  
#获取字段名列表
col_names=df.columns.values.tolist() 

x=df[col_names[0:-1]]
print(x.shape)
#取数据最后一列为标签列
y=df[col_names[-1]]

print(y.shape)
ss=StandardScaler()
X=ss.fit_transform(x)

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = 3))
    return(rmse)

#调用LassoCV函数，并进行交叉验证，默认cv=3
model_lasso = LassoCV(alphas = [0.1,1,0.001, 0.0005]).fit(X, y)

#模型所选择的最优正则化参数alpha
print(model_lasso.alpha_)

#各特征列的参数值或者说权重参数，为0代表该特征被模型剔除了
print(model_lasso.coef_)

#输出看模型最终选择了几个特征向量，剔除了几个特征向量
coef = pd.Series(model_lasso.coef_, index = col_names[0:-1])
print("Lasso选择的特征个数 " + str(sum(coef != 0)) + " 去掉的个数：variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
print('-----按重要度排序得到前100个特征的重要度和特征名----------')
data1=coef.sort_values(ascending=False)
print(data1[0:100])
print('-----按重要度排序得到前100个特征名------------')
print(data1[0:100].index)

'''
#输出所选择的最优正则化参数情况下的残差平均值，因为是3折，所以看平均值
print(rmse_cv(model_lasso).mean())


#画出特征变量的重要程度，这里面选出前10个重要，后10个重要的举例

imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
plt.show() 


reg = LassoCV(cv=5, random_state=0).fit(X, y)
score=reg.score(X, y) 
print(score)
'''