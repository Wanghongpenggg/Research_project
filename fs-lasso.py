import pandas as pd
import numpy as np
import csv as csv
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


#Feature selection using LassoCV

df = pd.read_csv("sd.csv",header=0)
from sklearn.utils import shuffle  
#Data sequence disruption
df = shuffle(df)  
#Get a list of names
col_names=df.columns.values.tolist() 
#print(col_names)
#x = df.drop(df["_RFHYPE5"])
#y = df["_RFHYPE5"]
#Take the first column of data to the first column of the last column as the data column
x=df[col_names[0:-1]]
print(x.shape)
#Take the last column of the data as the label column
y=df[col_names[-1]]

print(y.shape)
ss=StandardScaler()
X=ss.fit_transform(x)

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = 3))
    return(rmse)

#Call the LassoCV function and cross-validate, default cv=3
model_lasso = LassoCV(alphas = [0.1,1,0.001, 0.0005]).fit(X, y)

#The optimal regularization parameter alpha selected by the model
print(model_lasso.alpha_)

#The parameter value or the weight parameter of each feature column is 0, indicating that the feature is removed by the model.
print(model_lasso.coef_)

#The output looks at the model and finally selects several feature vectors, eliminating several feature vectors.
coef = pd.Series(model_lasso.coef_, index = col_names[0:-1])
print("Lasso选择的特征个数 " + str(sum(coef != 0)) + " 去掉的个数：variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

#Output the average of the residuals in the case of the selected optimal regularization parameter. Because it is 3 fold, look at the average.
print(rmse_cv(model_lasso).mean())


#Draw the importance of the characteristic variables, select the top 10 important, the last 10 important examples

imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
plt.show() 

'''
reg = LassoCV(cv=5, random_state=0).fit(X, y)
score=reg.score(X, y) 
print(score)
'''