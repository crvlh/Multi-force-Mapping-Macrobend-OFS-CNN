# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:16:32 2024

@author: vinic
"""
import numpy as np

print(f"\n TEST")

def mae_for_non_zero_values(y_true, y_pred):
    column_maes = []
    for col_idx in range(y_true.shape[1]):
        column_true = y_true[:, col_idx]
        column_pred = y_pred[:, col_idx]
        non_zero_indices = np.where((column_true != 0) | (column_pred != 0))
        column_true_non_zero = column_true[non_zero_indices]
        column_pred_non_zero = column_pred[non_zero_indices]
        column_mae = np.mean(np.abs(column_true_non_zero - column_pred_non_zero))
        column_maes.append(column_mae)
    overall_mae = np.mean(column_maes)
    return overall_mae, column_maes

# Example usage
overall_mae, column_maes = mae_for_non_zero_values(y_test, detect_test_reg)

def print_mae_results(overall_mae, column_maes):
    print("Average MAE:", "{:.2f}".format(overall_mae))
    print("MAE by Region:")
    for i, mae in enumerate(column_maes):
        print("Region", i + 1, ":", "{:.2f}".format(mae))
    mean_region_mae = np.mean(column_maes)
    std_region_mae = np.std(column_maes)
    print("Mean of Region MAEs:", "{:.2f}".format(mean_region_mae))
    print("Standard Deviation of Region MAEs:", "{:.2f}".format(std_region_mae))

# Example usage
print_mae_results(overall_mae, column_maes)

print("\nMultilabel Classification (Regions with Load):")

from sklearn.metrics import f1_score

# Convert to binary labels
y_test_bin = (y_test > 0).astype(int)
detect_test_bin = (detect_test_class > 0).astype(int)

# Calculating Weighted F1-score
f1_weighted = f1_score(y_test_bin, detect_test_bin, average='weighted')

print(f"Weighted F1-score: {f1_weighted:.2f}")

from sklearn.metrics import f1_score

# Convert to binary labels
y_test_bin = (y_test > 0).astype(int)
detect_test_bin = (detect_test_class > 0).astype(int)

# Calculating Macro F1-score
f1_macro = f1_score(y_test_bin, detect_test_bin, average='macro')

# Calculating Micro F1-score
f1_micro = f1_score(y_test_bin, detect_test_bin, average='micro')

print(f"Macro F1-score: {f1_macro:.2f}")
print(f"Micro F1-score: {f1_micro:.2f}")

def hamming_score(y_true, y_pred):
    correct_labels = np.sum(y_true == y_pred)
    total_labels = np.prod(y_true.shape)
    return correct_labels / total_labels

# Assuming y_test_bin contains the true labels and detect_test_bin contains the predicted labels
hamming_score_value = hamming_score(y_test_bin, detect_test_bin)

print(f"Hamming Score: {hamming_score_value}")

def custom_accuracy(detect_test_bin, y_test_bin):
    # Dictionary to map each position to its valid neighbors
    adjacency = {
        1: [1, 2, 5, 6],
        2: [1, 2, 3, 5, 6, 7],
        3: [2, 3, 4, 6, 7, 8],
        4: [3, 4, 7, 8],
        5: [1, 2, 5, 6, 9, 10],
        6: [1, 2, 3, 5, 6, 7, 9, 10, 11],
        7: [2, 3, 4, 6, 7, 8, 10, 11, 12],
        8: [3, 4, 7, 8, 11, 12],
        9: [5, 6, 9, 10, 13, 14],
        10: [5, 6, 7, 9, 10, 11, 13, 14, 15],
        11: [6, 7, 8, 10, 11, 12, 14, 15, 16],
        12: [7, 8, 11, 12, 15, 16],
        13: [9, 10, 13, 14],
        14: [9, 10, 11, 13, 14, 15],
        15: [10, 11, 12, 14, 15, 16],
        16: [11, 12, 15, 16]
    }
    
    correct_predictions = 0
    total_predictions = 0

    # Iterate over each sample
    for sample_idx in range(detect_test_bin.shape[0]):
        for pos_idx in range(detect_test_bin.shape[1]):
            # Check if it is a positive prediction
            if detect_test_bin[sample_idx, pos_idx] == 1:
                total_predictions += 1
                # Find the actual positive position in the same sample
                for target_pos in range(y_test_bin.shape[1]):
                    if y_test_bin[sample_idx, target_pos] == 1:
                        # Check if the predicted position is in the valid neighbors of the target
                        if (pos_idx + 1) in adjacency[target_pos + 1]:
                            correct_predictions += 1
                            break

    # Calculate accuracy based on the number of correct predictions
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy

print(f"Custom accuracy: {custom_accuracy(detect_test_bin, y_test_bin)}")
