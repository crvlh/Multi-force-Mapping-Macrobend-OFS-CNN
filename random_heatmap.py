# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:21:10 2024

@author: vinic
"""
import numpy as np
import matplotlib.pyplot as plt
import random

# Setting the seed to ensure reproducibility
random.seed(0)

# Selecting 5 random samples between 0 and 123
randomly_selected_samples = random.sample(range(124), 5)

# Loop to plot heatmaps for the selected samples
for i, sample_idx in enumerate(randomly_selected_samples):
    
    sample_detect = detect_test_class[sample_idx]
    sample_y = y_test[sample_idx]

    # Reshape samples to a 4x4 format
    sample_detect_reshaped = sample_detect.reshape((4, 4))
    sample_y_reshaped = sample_y.reshape((4, 4))

    # Plotting subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))

    # Plotting heatmap for detect_test
    axs[0].set_title(f"Sample {sample_idx + 1} - Detect", fontsize=22)
    im1 = axs[0].imshow(sample_detect_reshaped, cmap='YlGnBu', interpolation='nearest', extent=[0, 20, 0, 20], vmin=0.000, vmax=3000)

    # Adding numbers inside each square
    counter = 1
    for y in range(sample_detect_reshaped.shape[0]):
        for x in range(sample_detect_reshaped.shape[1]):
            axs[0].text(x * 5 + 2.5, (3 - y) * 5 + 2.5, f'{counter}', color='black', ha='center', va='center', fontsize=16)
            counter += 1

    axs[0].set_xlabel('X (cm)', fontsize=20)
    axs[0].set_ylabel('Y (cm)', fontsize=20)
    axs[0].set_xticks(np.arange(0, 21, 5))
    axs[0].set_yticks(np.arange(0, 21, 5))
    axs[0].set_xticklabels(np.arange(0, 21, 5), fontsize=16)
    axs[0].set_yticklabels(np.arange(0, 21, 5), fontsize=16)
    axs[0].grid(True)  # Enable background grid

    # Plotting heatmap for y_test
    axs[1].set_title(f"Sample {sample_idx + 1} - Target", fontsize=22)
    im2 = axs[1].imshow(sample_y_reshaped, cmap='YlGnBu', interpolation='nearest', extent=[0, 20, 0, 20], vmin=0.000, vmax=3000)

    # Adding numbers inside each square
    counter = 1
    for y in range(sample_y_reshaped.shape[0]):
        for x in range(sample_y_reshaped.shape[1]):
            axs[1].text(x * 5 + 2.5, (3 - y) * 5 + 2.5, f'{counter}', color='black', ha='center', va='center', fontsize=16)
            counter += 1

    axs[1].set_xlabel('X (cm)', fontsize=20)
    axs[1].set_ylabel('Y (cm)', fontsize=20)
    axs[1].set_xticks(np.arange(0, 21, 5))
    axs[1].set_yticks(np.arange(0, 21, 5))
    axs[1].set_xticklabels(np.arange(0, 21, 5), fontsize=16)
    axs[1].set_yticklabels(np.arange(0, 21, 5), fontsize=16)
    axs[1].grid(True)  # Enable background grid

    # Adding the colorbar to the second subplot
    cbar = fig.colorbar(im2, ax=axs[1])
    cbar.set_label('Magnitude values (gf)', fontsize=20)
    cbar.ax.tick_params(labelsize=20) 

    plt.tight_layout()
    plt.show()
