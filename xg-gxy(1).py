from xgboost import plot_importance
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from matplotlib import  pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# Load data set
df = pd.read_csv('sd.csv', header=0)
from sklearn.utils import shuffle  
df = shuffle(df)  
col_names=df.columns.values.tolist() 
#col_names=df.columns.values.tolist() 
X=df[col_names[0:-1]]
print(X.shape)
#Take out the Label column (whether it is sick)
Y=df[col_names[-1]]
print(Y.shape)
# Split the data set into training and test sets
seed = 7
test_size = 0.3

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=33)

# Fitting the XGBoost model

model = XGBClassifier(learning_rate=0.1,
                      n_estimators=1000,         # The number of trees - 1000 trees to establish xgboost
                      max_depth=6,               # Depth of the tree
                      min_child_weight = 1,      # Leaf node minimum weight
                      gamma=0.,                  # The parameter before the number of leaf nodes in the penalty item
                      subsample=0.8,             # Randomly select 80% samples to establish a decision tree
                      colsample_btree=0.8,       # Randomly select 80% feature to establish decision tree
                      objective='multi:softmax', # Specified loss function
                      scale_pos_weight=1,        # Solve the problem of unbalanced sample size
                      num_class=2,
                      random_state=27            # random number
                      )

model.fit(X_train, y_train,
          eval_set = [(X_test,y_test)],
          eval_metric = "mlogloss",
          early_stopping_rounds = 10,        
          verbose = True)

### plot feature importance
fig,ax = plt.subplots(figsize=(15,15))
plot_importance(model,
                height=0.5,
                ax=ax,
                max_num_features=64)
plt.show()

# Make predictions about test sets
y_pred = model.predict(X_test)

### model evaluate
accuracy = accuracy_score(y_test,y_pred)
print("accuarcy: %.2f%%" % (accuracy*100.0))
importance = model.feature_importances_
#print(importance)
data=pd.Series(importance,index=col_names[0:-1])
print('-----Sort the importance of the top 100 features by importance----------')
data1=data.sort_values(ascending=False)
print(data1[0:100])
print('-----Sort the top 100 features by importance------------')
print(data1[0:100].index)
'''
accuarcy: 77.72%
_BMI5       0.037542
BPEATHBT    0.031789
_AGE80      0.026945
WEIGHT2     0.026642
GENHLTH     0.025734
MAXVO2_     0.022101
CHOLMED1    0.019679
IDATE       0.018771
DIABAGE2    0.016197
IDAY        0.016046
_STSTR      0.016046
SEQNO       0.015743
_VEGESU1    0.014381
_STRWT      0.013624
WTKG3       0.013170
_LLCPWT     0.012867
_LLCPWT2    0.012564
PHYSHLTH    0.011807
CHECKUP1    0.010596
JOINPAI1    0.010294
FRUIT2      0.009840
_MICHD      0.009537
POTADA1_    0.009537
MARITAL     0.009234
FRENCHF1    0.009234
_WT2RAKE    0.009083
_DRNKWEK    0.009083
_FRUTSU1    0.008931
FVGREEN1    0.008780
CHOLCHK1    0.008629
              ...   
PADUR2_     0.004844
EXEROFT1    0.004693
DIFFWALK    0.004541
PREGNANT    0.004541
_AGEG5YR    0.004541
_CLLCPWT    0.004390
EXEROFT2    0.004390
PAFREQ2_    0.004239
STRFREQ_    0.004239
AVEDRNK2    0.004087
_CHISPNC    0.003936
INTERNET    0.003936
HTIN4       0.003936
HHADULT     0.003784
WTCHSALT    0.003784
PADUR1_     0.003784
_IMPRACE    0.003784
_RFCHOL1    0.003633
PDIABTST    0.003633
IMFVPLAC    0.003482
TOLDHI2     0.003482
METVL11_    0.003482
BLDSUGAR    0.003482
EXERHMM2    0.003330
CVDCRHD4    0.003330
PA1VIGM_    0.003330
DRNK3GE5    0.003330
SSBSUGR2    0.003028
MARIJANA    0.003028
DRVISITS    0.003028
Length: 100, dtype: float32
-----Sort the top 100 features by importance------------
Index(['_BMI5', 'BPEATHBT', '_AGE80', 'WEIGHT2', 'GENHLTH', 'MAXVO2_',
       'CHOLMED1', 'IDATE', 'DIABAGE2', 'IDAY', '_STSTR', 'SEQNO', '_VEGESU1',
       '_STRWT', 'WTKG3', '_LLCPWT', '_LLCPWT2', 'PHYSHLTH', 'CHECKUP1',
       'JOINPAI1', 'FRUIT2', '_MICHD', 'POTADA1_', 'MARITAL', 'FRENCHF1',
       '_WT2RAKE', '_DRNKWEK', '_FRUTSU1', 'FVGREEN1', 'CHOLCHK1', 'INCOME2',
       'GRENDA1_', 'VEGETAB2', 'POTATOE1', 'MENTHLTH', 'VEGEDA2_', 'EDUCA',
       'PERSDOC2', 'FRNCHDA_', 'DIABETE3', 'STRENGTH', '_DUALCOR', 'POORHLTH',
       'HEIGHT3', 'HIVTSTD3', 'MAXDRNKS', 'CHCKIDNY', 'ALCDAY5', 'FLSHTMY2',
       'EMPLOY1', 'CVDASPRN', '_STATE', 'PAMIN21_', 'METVL21_', '_RFBMI5',
       '_PRACE1', 'FRUITJU2', 'PNEUVAC3', '_PHYS14D', 'FMONTH', 'PAFREQ1_',
       'FTJUDA2_', 'FRUTDA2_', 'EXERHMM1', 'PA1MIN_', 'DROCDY3_', '_MINAC11',
       '_MINAC21', 'EXRACT21', 'EXRACT11', 'PADUR2_', 'EXEROFT1', 'DIFFWALK',
       'PREGNANT', '_AGEG5YR', '_CLLCPWT', 'EXEROFT2', 'PAFREQ2_', 'STRFREQ_',
       'AVEDRNK2', '_CHISPNC', 'INTERNET', 'HTIN4', 'HHADULT', 'WTCHSALT',
       'PADUR1_', '_IMPRACE', '_RFCHOL1', 'PDIABTST', 'IMFVPLAC', 'TOLDHI2',
       'METVL11_', 'BLDSUGAR', 'EXERHMM2', 'CVDCRHD4', 'PA1VIGM_', 'DRNK3GE5',
       'SSBSUGR2', 'MARIJANA', 'DRVISITS'],
      dtype='object')


'''