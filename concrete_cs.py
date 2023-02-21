# -*- coding: utf-8 -*-
"""concrete_cs.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Y7encYzevoe7m14Ux_rT-PkJtX1jhCau
"""

# Commented out IPython magic to ensure Python compatibility.
# ANN Modelling in TensorFlow and Keras

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout

# Data Manipulation
import numpy as np
import pandas as pd

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
sns.set()
import pickle
# Data Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Model Evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score

"""Data Preprocessing"""

#from google.colab import drive
#drive.mount('/content/drive')

#path = "/content/drive/MyDrive/concrete_data.csv"

"""Import and Check Data"""

concrete_data = pd.read_csv("concrete_data.csv")

concrete_data.head()

"""Train Test Split """

x = concrete_data.drop('concrete_compressive_strength',axis = 1)
y = concrete_data['concrete_compressive_strength']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 46)

"""Scale the Data"""

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

"""ANN Model"""

x_train.shape

model = Sequential()

# We will start with 45 hidden layers. 
model.add(Dense(8,activation='relu'))
 # All layers utilize rectified linear units (relu)
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse') # Use the adam optimization algorithm

"""Training the model"""

model.fit(x=x_train,y=y_train.values,
          validation_data=(x_test,y_test.values),
          batch_size=128,epochs=200)

"""Visualize the Loss Function"""

losses = pd.DataFrame(model.history.history)
losses.plot()

"""Testing the Model"""

predictions = model.predict(x_test)

"""Model Evaluation"""

# Model Evaluation Metrics
MAE = mean_absolute_error(y_test,predictions)
RMSE = np.sqrt(mean_squared_error(y_test,predictions))                 
EVS = explained_variance_score(y_test,predictions)

print('EVALUATION METRICS')
print('-----------------------------')
print(f"Mean Absolute Error (MAE):\t\t{MAE}\nRoot Mean Squared Error (RMSE):\t\t{RMSE}\nExplained Variance Score:\t\t{EVS}")

pickle.dump(model, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))

