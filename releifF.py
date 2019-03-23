import pandas as pd
import numpy as np
import ReliefF as rf
from pandas import DataFrame
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('sd.csv', header=0)
from sklearn.utils import shuffle
#Data sequence disruption
df = shuffle(df)
col_names=df.columns.values.tolist()
X=df[col_names[0:-1]]
names = X.columns.values.tolist()
ss=StandardScaler()
X=ss.fit_transform(X)
print(X.shape)
X_ = np.array(X)
Y=df[col_names[-1]]
Y_ = np.array(Y)

print(names)
#print(X)
#print(Y)

r = rf.ReliefF(n_neighbors=10, n_features_to_keep=10)

r_fit = r.fit(X_,Y_)

#print(r.feature_scores)

coef = DataFrame(r.feature_scores, index=names)
coef = pd.Series(coef)
print(coef)


imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
print(imp_coef)
'''
print(imp_coef)
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
coef.plot(kind = "barh")
plt.title("Coefficients in the relieff Model")
plt.show()
'''


