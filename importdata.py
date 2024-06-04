# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:05:08 2024

@author: vinic
"""

import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

########## DATA IMPORT ##########

# Target import

# Path to the file
data_folder1 = "C:\\Users\\vinic\\Desktop\\Doutorado\\TESE 2\\Dados\\dados vinicius\\target.xlsx"
# Read the Excel spreadsheet into a pandas DataFrame
df = pd.read_excel(data_folder1, header=0)    
targets_reg = df.values

# Input import

# Define the path to the folder containing optical data files
data_folder2 = "C:\\Users\\vinic\\Desktop\\Doutorado\\TESE 2\\Dados\\dados vinicius" 
# Filter
n_wavelength_effective = 1734 # Number of wavelengths
skip_rows = 707  # Row index of the first considered wavelength 
# Import files (multiple files, each file a sample), read spectral data, filter wavelengths of interest.
file_list = glob.glob(os.path.join(data_folder2, "*.txt"))
file_list = sorted(file_list, key=lambda x: int(os.path.basename(x).split(".")[0]))
num_samples = len(file_list)
# Read spectral data from each file and reshape into a 2D array
spect_data = [pd.read_csv(file, delimiter='\t', skiprows=skip_rows, nrows=n_wavelength_effective, usecols=range(1, 2))
                      for file in file_list]
spect_data = np.reshape(np.array(spect_data), (num_samples, n_wavelength_effective))
# Read wavelengths from the first file
wavelengths = pd.read_csv(file_list[0], sep='\t', header=None, usecols=[0], skiprows=(skip_rows + 1), nrows=n_wavelength_effective, index_col=None)

########## END OF DATA IMPORT ##########

# Create the figure and axes
fig, ax = plt.subplots(figsize=(10, 5))

# Spectral data of training and validation
ax.plot(wavelengths, np.transpose(spect_data), linewidth=1.0)
ax.tick_params(direction='in')
ax.set_title("Spectral curves")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Relative Transmitted Intensity(%)") # Your optical parameter (reflectance, transmittance, scope, absorvance)
ax.set_xlim(380, 830)
ax.set_ylim(-10, 110)

plt.show()

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(spect_data, targets_reg, test_size=0.20, random_state=0)

from sklearn.preprocessing import StandardScaler

# Scaling the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train = X_train_scaled
X_test = X_test_scaled
