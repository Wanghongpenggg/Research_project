import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("sample_data.csv")
frames = [df["MAXVO2_"], df["CHILDREN"], df["HEIGHT3"], df["_AGE80"], df["NOBCUSE6"], df["_FRUTSU1"], df["_BMI5"],df["WEIGHT2"], df["FRUTDA2_"]]
X = pd.concat(frames, axis=1)
y = df["_RFHYPE5"]


xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.30, random_state=531)

yTrain = np_utils.to_categorical(yTrain)
yTest = np_utils.to_categorical(yTest)

model = Sequential([
    Dense(20, input_dim=9),
    Activation('tanh'),
    Dense(20, input_dim=9),
    Activation('tanh'),
    Dense(20, input_dim=9),
    Activation('tanh'),
    Dense(20, input_dim=9),
    Activation('tanh'),
    Dense(20, input_dim=9),
    Activation('tanh'),
    Dense(20, input_dim=9),
    Activation('tanh'),
    Dense(20, input_dim=9),
    Activation('tanh'),
    Dense(20, input_dim=9),
    Activation('tanh'),
    Dense(20, input_dim=9),
    Activation('tanh'),
    Dense(20, input_dim=9),
    Activation('tanh'),
    Dense(3),
    Activation('softmax'),
])

# rmsprop = RMSprop(lr=0.001, rho=0.8, epsilon=None, decay=0.0)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(xTrain, yTrain, epochs=100, batch_size=1000)

score = model.evaluate(xTest, yTest, batch_size=100)

print(score)




