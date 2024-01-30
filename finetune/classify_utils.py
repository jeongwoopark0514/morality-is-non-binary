import json
import torch
import pandas as pd
import random
import numpy as np
import statistics
import sklearn.metrics as metrics
from collections import defaultdict
from decimal import Decimal

def set_seeds(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

# Summarize the f1 results in a table

def f1_results(f1_scores, labels):
    values = np.array([[label, round(Decimal(statistics.mean(f1_scores[label])), 3),
                        round(Decimal(statistics.stdev(f1_scores[label])), 3)] for label in
                       labels])
    classification_table = pd.DataFrame(values, columns=['Labels', 'Mean', 'SD'])
    print(classification_table)
    mean_f1 = round(Decimal(statistics.mean(classification_table["Mean"])), 2)
    return classification_table, mean_f1


# Print all results of the experiment.
def print_results(f1_scores, labels, filename):
    classification_table, f1_source = f1_results(f1_scores, labels)

    f = open(f"result-{filename}.txt", "a")
    f.write(f'Table: {classification_table}')
    f.write(f'\nF1 Scores: {f1_scores}')
    f.write(f'\nAverage F1: {f1_source}')
    if len(labels) > 2:
        f.write(f'\nAverage Micro F1 Source: {statistics.mean(f1_scores["micro avg"])}')
    else:
        f.write(f'\nAverage Accuracy Source: {statistics.mean(f1_scores["accuracy"])}')
    f.write(f'\n Average Macro F1 Source: {statistics.mean(f1_scores["macro avg"])}')
    f.write(f'\n Average Weighted F1 Source: {statistics.mean(f1_scores["weighted avg"])}')
    f.close()

    print(f'\nAverage F1: {f1_source}')

    if len(labels) > 2:
        print(f'\nAverage Micro F1 Source: {statistics.mean(f1_scores["micro avg"])}')
    else:
        print(f'\nAverage Accuracy: {statistics.mean(f1_scores["accuracy"])}')
    print(f'Average Macro F1 Source: {statistics.mean(f1_scores["macro avg"])}')
    print(f'Average Weighted F1 Source: {statistics.mean(f1_scores["weighted avg"])}')


# No longer used
# def regression_mse_labels(mse, labels):
#     values = np.array([[label, round(Decimal(statistics.mean(mse[label])), 2),
#                         round(Decimal(statistics.stdev(mse[label])), 2)] for label in
#                        labels])
#     regression_table = pd.DataFrame(values, columns=['Labels', 'Mean', 'SD'])
#     print(regression_table)
#     mean_f1 = round(Decimal(statistics.mean(regression_table["Mean"])), 2)
#     return regression_table, mean_f1
#
#
# def regression_results(y_true, y_pred):
#     # Regression metrics
#     explained_variance=metrics.explained_variance_score(y_true, y_pred)
#     mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred)
#     mse=metrics.mean_squared_error(y_true, y_pred)
#     # mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
#     median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
#     r2=metrics.r2_score(y_true, y_pred)
#
#     # MSE per label
#     num_instances, labels = y_true.shape
#     print("Num instances: ", num_instances)
#     print("Labels: ", labels)
#     mse_labels = []
#     for i in range(labels):
#         mse_labels.append(metrics.mean_squared_error(y_true[:, i], y_pred[:, i]))
#
#     print('explained_variance: ', round(explained_variance,4))
#     # print('mean_squared_log_error: ', round(mean_squared_log_error,4))
#     print('r2: ', round(r2,4))
#     print('MAE: ', round(mean_absolute_error,4))
#     print('MSE: ', round(mse,4))
#     print('RMSE: ', round(np.sqrt(mse),4))
#
#     regression_result_dict = {
#         "explained_variance" : explained_variance,
#         "mean_absolute_error" : mean_absolute_error,
#         "mse" : mse,
#         "median_absolute_error" : median_absolute_error,
#         "r2" : r2,
#         "mse_labels": mse_labels
#     }
#     return regression_result_dict
#