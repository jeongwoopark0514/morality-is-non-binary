import statistics
import random
from decimal import Decimal
from sklearn.metrics import classification_report

import torch
import pandas as pd
import numpy as np

datasets = ['ALM', 'Baltimore', 'BLM', 'Davidson', 'Election', 'MeToo', 'Sandy']


def set_seeds(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


# Remove left_out from the datasets array, read each csv in a dataframe and create the datasets.
def combine_datasets(left_out, target_frac=1):
    train_corpora = [x for x in datasets if x != left_out]
    train_data = pd.concat([pd.read_csv(f'nlp/data/processed/{f}.csv') for f in train_corpora],
                           ignore_index=True).sample(frac=1)
    test_data = pd.read_csv(f'nlp/data/processed/{left_out}.csv').sample(frac=target_frac)
    return train_data, test_data


def get_dataset(datasets):
    data = pd.concat([pd.read_csv(f'nlp/data/processed/{f}.csv') for f in datasets],
                     ignore_index=True).sample(frac=1)
    return data


# Create a classification report and return a list of json objects with predictions.
def classification(model, dataset, labels):
    y_predicted, y_true = model.predict(dataset)
    print(classification_report(y_true, y_predicted, target_names=labels, digits=3), '\n')
    objects = make_objects(dataset.text, y_predicted, y_true, labels)
    return classification_report(y_true, y_predicted, target_names=labels, output_dict=True, zero_division=0), objects


# Create a list of json objects with predictions
def make_objects(texts, y_predicted, y_true, labels):
    list_of_objects = []
    for i in range(len(texts)):
        predicted_labels = [labels[j] for j, x in enumerate(y_predicted[i]) if x == 1]
        true_labels = [labels[j] for j, x in enumerate(y_true[i]) if x == 1]
        list_of_objects.append(
            {
                'text': texts[i],
                'predicted': predicted_labels,
                'actual': true_labels
            }
        )
    return list_of_objects


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
def print_results(f1_scores, labels, file_name):
    classification_table, f1_source = f1_results(f1_scores, labels)

    f = open(file_name, "a")
    f.write(f'Table: {classification_table}')
    f.write(f'\nF1 Scores: {f1_scores}')
    f.write(f'\nAverage F1: {f1_source}')
    f.write(f'\nAverage Micro F1 Source: {statistics.mean(f1_scores["micro avg"])}')
    f.write(f'\n Average Macro F1 Source: {statistics.mean(f1_scores["macro avg"])}')
    f.write(f'\n Average Weighted F1 Source: {statistics.mean(f1_scores["weighted avg"])}')
    f.close()

    print(f'\nAverage F1: {f1_source}')

    print(f'\nAverage Micro F1 Source: {statistics.mean(f1_scores["micro avg"])}')
    print(f'Average Macro F1 Source: {statistics.mean(f1_scores["macro avg"])}')
    print(f'Average Weighted F1 Source: {statistics.mean(f1_scores["weighted avg"])}')