# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:07:08 2024

@author: vinic
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
import random

# Set seed
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

# KFold configuration
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
fold_no = 1
loss_per_fold = []
mae_per_fold = []
epochs_list = []

for train_index, test_index in kf.split(X_train):
    X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
    y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
    
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
        Dense(16, activation='linear')  
    ])

    model.compile(loss=huber_loss, optimizer='adam', metrics=['mae', 'mse'])
    
    early_stopping = EarlyStopping(
        monitor='val_loss',  
        patience=10,         # Number of epochs with no improvement after which training will be stopped
        verbose=1,         
        restore_best_weights=True  
    )

    print(f'Training fold {fold_no}...')
    history = model.fit(X_train_fold, y_train_fold, batch_size=32, epochs=100, validation_data=(X_test_fold, y_test_fold), verbose=0, callbacks=[early_stopping])
    
    epochs_list.append(len(history.history['loss']))

    # Evaluate the model
    scores = model.evaluate(X_test_fold, y_test_fold, verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]}')
    loss_per_fold.append(scores[0])
    mae_per_fold.append(scores[1])
    
    # Plot training and validation MAE curves
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['mae'], 'b', label='Training MAE')
    plt.plot(history.history['val_mae'], 'r', label='Validation MAE')
    plt.title(f'Fold {fold_no} Validation MAE') 
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.ylim([0, 300])
    plt.legend()
    plt.show()

    fold_no += 1
    
    effective_epochs = len(history.history['loss'])
    print(f'Training completed after {effective_epochs} epochs')

# Present average results
print(f'Average Loss: {np.mean(loss_per_fold)}')
print(f'Average MAE: {np.mean(mae_per_fold)}')
print(f'Standard deviation of MAE across folds: {np.std(mae_per_fold)}')
# Calculate the average number of epochs
average_epochs = int(np.mean(epochs_list))
print(f'Average epochs before Early Stopping: {average_epochs}')

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, n_splits + 1), mae_per_fold, 'r', label='MAE per Fold')
plt.xlabel('Fold')
plt.ylabel('Metrics')
plt.ylim([0, 1000])
plt.legend()
plt.title('Performance across Folds')
plt.show()
