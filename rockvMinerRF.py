from math import sqrt, fabs, exp
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import roc_auc_score, roc_curve
import numpy
import pandas as pd



'''
#read data from uci data repository
data=open("data\sonar.all-data")

line=data.readline()


#arrange data into list for labels and list of lists for attributes
xList = []

while line!="":
    #split on comma
    row = line.strip().split(",")
    xList.append(row)
    line=data.readline()

#separate labels from attributes, convert from attributes from string to numeric and convert "M" to 1 and "R" to 0

xNum = []
labels = []

for row in xList:
    lastCol = row.pop()
    if lastCol == "M":
        labels.append(1)
    else:
        labels.append(0)
    attrRow = [float(elt) for elt in row]
    xNum.append(attrRow)

#number of rows and columns in x matrix
nrows = len(xNum)#行数
ncols = len(xNum[1])#列数

#form x and y into numpy arrays and make up column names
X = numpy.array(xNum)
y = numpy.array(labels)
rocksVMinesNames = numpy.array(['V' + str(i) for i in range(ncols)])#写成V几
'''
df = pd.read_csv("sample_data.csv")#All data in the small data set
df0 = df[:10000]#All 1 data
df1 = df[10000:11000]#All 0 data
frames = [df["MAXVO2_"], df["CHILDREN"], df["HEIGHT3"], df["_AGE80"], df["NOBCUSE6"], df["_FRUTSU1"], df["_BMI5"],df["WEIGHT2"], df["FRUTDA2_"]]
frames0 = [df0["MAXVO2_"], df0["CHILDREN"], df0["HEIGHT3"], df0["_AGE80"], df0["NOBCUSE6"], df0["_FRUTSU1"], df0["_BMI5"],df0["WEIGHT2"], df0["FRUTDA2_"]]
frames1 = [df1["MAXVO2_"], df1["CHILDREN"], df1["HEIGHT3"], df1["_AGE80"], df1["NOBCUSE6"], df1["_FRUTSU1"], df1["_BMI5"],df1["WEIGHT2"], df1["FRUTDA2_"]]
X = pd.concat(frames, axis=1)
y = df["LABEL"]
X0 = pd.concat(frames0, axis=1)
y0 = df0["LABEL"]
X1 = pd.concat(frames1, axis=1)
y1 = df1["LABEL"]
#smo = SMOTE(random_state=42)
#X_smo, y_smo = smo.fit_sample(X, y)#Process all data with SMOTE
xTrain0, xTest0, yTrain0, yTest0 = train_test_split(X0, y0, test_size=0.30, random_state=531)
xTrain1, xTest1, yTrain1, yTest1 = train_test_split(X1, y1, test_size=0.30, random_state=531)
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.30, random_state=531)



#break into training and test sets.
#xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.30, random_state=531)

auc = []
nTreeList = range(50, 2000, 50)
for iTrees in nTreeList:
    depth = None
    maxFeat  = 8 #try tweaking
    rocksVMinesRFModel = ensemble.RandomForestClassifier(n_estimators=iTrees, max_depth=depth, max_features=maxFeat,
                                                 oob_score=False, random_state=531)

    rocksVMinesRFModel.fit(xTrain,yTrain)

    #Accumulate auc on test set
    prediction = rocksVMinesRFModel.predict_proba(xTest)
    aucCalc = roc_auc_score(yTest, prediction[:,1:2])
    auc.append(aucCalc)

print("AUC" )
print(auc[-1])


#plot training and test errors vs number of trees in ensemble
plot.plot(nTreeList, auc)
plot.xlabel('Number of Trees in Ensemble')
plot.ylabel('Area Under ROC Curve - AUC')
#plot.ylim([0.0, 1.1*max(mseOob)])
plot.show()

# Plot feature importance
featureImportance = rocksVMinesRFModel.feature_importances_

# normalize by max importance
featureImportance = featureImportance / featureImportance.max()

#plot importance of top 30
idxSorted = numpy.argsort(featureImportance)[30:60]
idxTemp = numpy.argsort(featureImportance)[::-1]
print(idxTemp)
barPos = numpy.arange(idxSorted.shape[0]) + .5
plot.barh(barPos, featureImportance[idxSorted], align='center')
plot.yticks(barPos, rocksVMinesNames[idxSorted])
plot.xlabel('Variable Importance')
plot.show()

#plot best version of ROC curve
fpr, tpr, thresh = roc_curve(yTest, list(prediction[:,1:2]))
ctClass = [i*0.01 for i in range(101)]

plot.plot(fpr, tpr, linewidth=2)
plot.plot(ctClass, ctClass, linestyle=':')
plot.xlabel('False Positive Rate')
plot.ylabel('True Positive Rate')
plot.show()

#pick some threshold values and calc confusion matrix for best predictions
#notice that GBM predictions don't fall in range of (0, 1)
#pick threshold values at 25th, 50th and 75th percentiles
idx25 = int(len(thresh) * 0.25)
idx50 = int(len(thresh) * 0.50)
idx75 = int(len(thresh) * 0.75)

#calculate total points, total positives and total negatives
totalPts = len(yTest)
P = sum(yTest)
N = totalPts - P

print('')
print('Confusion Matrices for Different Threshold Values')

#25th
TP = tpr[idx25] * P; FN = P - TP; FP = fpr[idx25] * N; TN = N - FP
print('')
print('Threshold Value =   ', thresh[idx25])
print('TP = ', TP/totalPts, 'FP = ', FP/totalPts)
print('FN = ', FN/totalPts, 'TN = ', TN/totalPts)

#50th
TP = tpr[idx50] * P; FN = P - TP; FP = fpr[idx50] * N; TN = N - FP
print('')
print('Threshold Value =   ', thresh[idx50])
print('TP = ', TP/totalPts, 'FP = ', FP/totalPts)
print('FN = ', FN/totalPts, 'TN = ', TN/totalPts)

#75th
TP = tpr[idx75] * P; FN = P - TP; FP = fpr[idx75] * N; TN = N - FP
print('')
print('Threshold Value =   ', thresh[idx75])
print('TP = ', TP/totalPts, 'FP = ', FP/totalPts)
print('FN = ', FN/totalPts, 'TN = ', TN/totalPts)


# Printed Output:
#
# AUC
# 0.950304259635
#
# Confusion Matrices for Different Threshold Values
#
# ('Threshold Value =   ', 0.76051282051282054)
# ('TP = ', 0.25396825396825395, 'FP = ', 0.0)
# ('FN = ', 0.2857142857142857, 'TN = ', 0.46031746031746029)
#
# ('Threshold Value =   ', 0.62461538461538457)
# ('TP = ', 0.46031746031746029, 'FP = ', 0.047619047619047616)
# ('FN = ', 0.079365079365079361, 'TN = ', 0.41269841269841268)
#
# ('Threshold Value =   ', 0.46564102564102566)
# ('TP = ', 0.53968253968253965, 'FP = ', 0.22222222222222221)
# ('FN = ', 0.0, 'TN = ', 0.23809523809523808)