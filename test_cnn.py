# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:15:05 2024

@author: vinic
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import random
import time
np.random.seed(0)
tf.random.set_seed(0)
random.seed(0)

def huber_loss(y_true, y_pred, delta=250):
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= delta
    small_error_loss = 0.5 * tf.square(error)
    large_error_loss = delta * (tf.abs(error) - 0.5 * delta)
    
    huber_loss = tf.where(is_small_error, small_error_loss, large_error_loss)
    return tf.reduce_mean(huber_loss)

# Model
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(1734, 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=3, activation='linear'),
    MaxPooling1D(pool_size=2),
    Conv1D(128, kernel_size=3, activation='linear'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='linear')  # Linear output for regression
])

# Model summary
model.summary()

model.compile(loss=huber_loss, optimizer='adam', metrics=['mse', 'mae'])

# Training 
start_time = time.time()

history = model.fit(X_train, y_train, batch_size=32, epochs=53, validation_data=(X_test, y_test)) #53epchs

end_time = time.time()
# Time
training_time = end_time - start_time
print(f'Training Time: {training_time:.2f} seconds')
print(history.history)

# Evaluating the model on the test set
loss, mse, mae = model.evaluate(X_test, y_test)
# print('Test Loss:', loss)
# print('Test MSE:', mse)
print('Test MAE:', mae)

# Making predictions on the test set
predictions = model.predict(X_test)

detect_test_reg = predictions

detect_test_class = np.where(predictions < 500, 0, predictions)
