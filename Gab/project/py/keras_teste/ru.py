from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
import itertools
np.random.seed(7)

train = np.loadtxt("TrainDatasetFinal.txt", delimiter=",")
test = np.loadtxt("testDatasetFinal.txt", delimiter=",")

y_train = train[:,7]
y_test = test[:,7]

magnitude_training = train[:,5]
norm_train = (magnitude_training - np.mean(magnitude_training))/np.std(magnitude_training)
magnitude_testing = test[:,5]
norm_test = (magnitude_testing - np.mean(magnitude_testing))/np.std(magnitude_testing)

model = Sequential()
model.add(Dense(12, input_dim=1, activation='relu'))
model.add(Dense(8,  activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy' , optimizer='adam', metrics=['accuracy'])

model.fit(norm_train, y_train, epochs=2, batch_size=64, verbose=2)

score=model.evaluate(norm_test, y_test, verbose=2)
print(score)