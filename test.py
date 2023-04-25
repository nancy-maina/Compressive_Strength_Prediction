import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import normalize

model = tf.keras.models.load_model('model_data/my_model.h5', compile= False)
model.load_weights('model_data/my_model_weights.h5')
model.compile()

# final_features = [[5, 10, 0, 0.5, 0, 10, 10, 28]]
final_features = [[0, 0, 0, 0, 0, 0, 0, 0]]

#  Normalize the data to be used in the model between 0 and 1
final_features = normalize(final_features)

# Prediction
prediction = model.predict(final_features)
print(prediction)
