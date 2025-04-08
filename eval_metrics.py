# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:16:01 2024

@author: vinic
"""

from sklearn.metrics import f1_score
import numpy as np

# Convert to binary labels
y_test_bin = (y_test > 0).astype(int)
detect_test_bin = (detect_test_class > 0).astype(int)

# Calculate F1 score and Hamming score for each column
num_columns = y_test_bin.shape[1]
f1_micro_scores = []
f1_macro_scores = []
f1_weighted_scores = []
hamming_scores = []

for col in range(num_columns):
    y_true_col = y_test_bin[:, col]
    y_pred_col = detect_test_bin[:, col]

    f1_micro = f1_score(y_true_col, y_pred_col, average='micro')
    f1_macro = f1_score(y_true_col, y_pred_col, average='macro')
    f1_weighted = f1_score(y_true_col, y_pred_col, average='weighted')

    f1_micro_scores.append(f1_micro)
    f1_macro_scores.append(f1_macro)
    f1_weighted_scores.append(f1_weighted)

    correct_labels = np.sum(y_true_col == y_pred_col)
    total_labels = len(y_true_col)
    hamming_score = correct_labels / total_labels

    hamming_scores.append(hamming_score)

for col in range(num_columns):
    print(f"Column {col + 1}:")
    print(f"\tF1 Micro: {f1_micro_scores[col]}")
    print(f"\tF1 Macro: {f1_macro_scores[col]}")
    print(f"\tF1 Weighted: {f1_weighted_scores[col]}")
    print(f"\tHamming Score: {hamming_scores[col]}")
    print()

from sklearn.metrics import mean_absolute_error

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


print_mae_results(overall_mae, column_maes)

# Calculate and print average F1 scores and Hamming score
avg_f1_micro = np.mean(f1_micro_scores)
avg_f1_macro = np.mean(f1_macro_scores)
avg_f1_weighted = np.mean(f1_weighted_scores)
avg_hamming_score = np.mean(hamming_scores)

print("\nAverage Scores Across All Columns:")
print(f"F1 Micro (Average): {avg_f1_micro:.4f}")
print(f"F1 Macro (Average): {avg_f1_macro:.4f}")
print(f"F1 Weighted (Average): {avg_f1_weighted:.4f}")
print(f"Hamming Score (Average): {avg_hamming_score:.4f}")

# Confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

y_test_concat = np.concatenate(y_test_bin, axis=None)
detect_test_concat = np.concatenate(detect_test_bin, axis=None)
conf_matrix = confusion_matrix(y_test_concat, detect_test_concat)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16})
plt.xlabel('Prediction', fontsize=16)
plt.ylabel('True', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()
