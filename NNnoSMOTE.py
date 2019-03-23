import numpy as np
from imblearn.over_sampling import SMOTE
np.random.seed(1337)  # for reproducibility
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("sample_data.csv")
df0 = df[:10000]
df1 = df[10000:11000]
frames = [df["MAXVO2_"], df["CHILDREN"], df["HEIGHT3"], df["_AGE80"], df["NOBCUSE6"], df["_FRUTSU1"], df["_BMI5"],df["WEIGHT2"], df["FRUTDA2_"]]
frames0 = [df0["MAXVO2_"], df0["CHILDREN"], df0["HEIGHT3"], df0["_AGE80"], df0["NOBCUSE6"], df0["_FRUTSU1"], df0["_BMI5"],df0["WEIGHT2"], df0["FRUTDA2_"]]#直觉手动选的一些属性
frames1 = [df1["MAXVO2_"], df1["CHILDREN"], df1["HEIGHT3"], df1["_AGE80"], df1["NOBCUSE6"], df1["_FRUTSU1"], df1["_BMI5"],df1["WEIGHT2"], df1["FRUTDA2_"]]
X = pd.concat(frames, axis=1)
y = df["LABEL"]
X0 = pd.concat(frames0, axis=1)
y0 = df0["LABEL"]
X1 = pd.concat(frames1, axis=1)
y1 = df1["LABEL"]
smo = SMOTE(random_state=42)
X_smo, y_smo = smo.fit_sample(X, y)#用SMOTE处理所有的数据
xTrain0, xTest0, yTrain0, yTest0 = train_test_split(X0, y0, test_size=0.30, random_state=531)
xTrain1, xTest1, yTrain1, yTest1 = train_test_split(X1, y1, test_size=0.30, random_state=531)
y_smo = np_utils.to_categorical(y_smo)
xTrain = np.concatenate([xTrain0,xTrain1])
yTrain = np.concatenate([yTrain0,yTrain1])
yTrain = np_utils.to_categorical(yTrain)
yTest0 = np_utils.to_categorical(yTest0,num_classes=2)
yTest1 = np_utils.to_categorical(yTest1,num_classes=2)
print(yTest0.shape)

model = Sequential([#建立神经网络模型
    Dense(32, input_dim=9),
    Activation('relu'),
    Dense(32, input_dim=9),
    Activation('relu'),
    Dense(32, input_dim=9),
    Activation('relu'),
    Dense(32, input_dim=9),
    Activation('relu'),
    Dense(32, input_dim=9),
    Activation('relu'),
    Dense(32, input_dim=9),
    Activation('relu'),
    Dense(32, input_dim=9),
    Activation('relu'),
    Dense(32, input_dim=9),
    Activation('relu'),
    Dense(32, input_dim=9),
    Activation('relu'),
    Dense(32, input_dim=9),
    Activation('relu'),
    Dense(32, input_dim=9),
    Activation('relu'),
    Dense(32, input_dim=9),
    Activation('relu'),
    Dense(2),
    Activation('softmax'),
])
model.compile(optimizer='rmsprop',#优化方法，用rmsprop和adam效果最好，其实用什么都一样，就是速度的问题
              loss='mean_squared_error',
              metrics=['accuracy'])
model.fit(xTrain, yTrain, epochs=50, batch_size=1000)#把补全了的数据用来训练模型
score0 = model.evaluate(xTest0, yTest0, batch_size=10)
score1 = model.evaluate(xTest1, yTest1, batch_size=10)
print(score0)
print(score1)