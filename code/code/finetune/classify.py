import dataset
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim, sched
from transformers import get_linear_schedule_with_warmup
import time, datetime

from classifiers import LinearCLS
from classify_utils import set_seeds, print_results
from collections import defaultdict
from scipy.special import expit
from simcse import SimCSE
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader


datasets = ["ALM", "Baltimore", "BLM", "Davidson", "Election", "MeToo", "Sandy"]

mftc_moral_values = ["fairness", "non-moral", "purity", "degradation", "loyalty", "care", "cheating", "betrayal",
               "subversion", "authority", "harm"]

moral_foundations = ['care', 'fairness', 'loyalty', 'authority', 'purity', 'non-moral']

# Please input appropriate names for each parameter.
default_config = {
    "simcse_model": "model/unsup-large-foundation",
    # "simcse_model": "model/sup-batch16/large-lr5e-05-ep3-seq64-batch16-temp0.05",
    "data_file": "data/merged_MFTC_test_add.csv",
    "input_size": 1024,
    "max_length": 64,
    "label_names": moral_foundations,
    "use_foundations": True,
    "epochs": 10,
    "loss_fct": nn.BCEWithLogitsLoss(),
    "threshold": 0.5,
    "batch_size": 16,
    "learning_rate": 0.01,
    "dropout": 0.1,
    "save_path": "classifiers/classifier_test.pt",
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
}

ablation_config = {
    "simcse_model": ["model/new-unsup-batch16/unsup-large-lr3e-05-ep1-seq64-batch16-temp0.05", "model/large-foundation-supervised-simcse-setting"],
    "data_file": "data/merged_MFTC_test_add.csv",
    # "data_file": ["data/test_set1.csv", "data/test_set2.csv", "data/test_set3.csv", "data/test_set4.csv", "data/test_set5.csv"],
    "input_size": 1024,
    "max_length": 64,
    "label_names": moral_foundations,
    "use_foundations": True,
    "epochs": 10,
    "loss_fct": nn.BCEWithLogitsLoss(),
    "threshold": 0.5,
    "batch_size": 16,
    "learning_rate": 0.01,
    "dropout": 0.1,
    "save_path": "classifiers/classifier_test.pt",
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
}


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train(model, train_set, config):
    model.train()
    train_set.to(config['device'])
    train_loader = DataLoader(dataset=train_set, batch_size=config.get("batch_size"), shuffle=True)
    criterion = config['loss_fct']
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    epoch_t0 = time.time()
    total_steps = len(train_loader) * config['epochs']
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    for i in range(config['epochs']):
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad() # Set all gradient values to zeros.
            b_input_ids = batch["encodings"]
            b_labels = batch["labels"]
            y_pred = model(b_input_ids)
            loss = criterion(y_pred, b_labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # scheduler.step()

    print(f'Total training time: {format_time(time.time() - epoch_t0)}')
    return model


def evaluate(model, test_loader, config):
    model.eval()
    y_predicted = []
    y_true = []
    total_loss = 0
    criterion = config['loss_fct']

    for step, batch in enumerate(test_loader):
        with torch.no_grad():
            b_input_ids = batch["encodings"]
            b_labels = batch["labels"]
            y_pred = model(b_input_ids)
            loss = criterion(y_pred, b_labels.float())
            total_loss += loss.item()
        y_predicted.extend(y_pred)
        y_true.extend(b_labels)

    return torch.stack(y_predicted), torch.stack(y_true),  total_loss/len(test_loader)


def predict(model, dataset, config, save = False):
    # model.eval()
    dataset.to(config['device'])
    loader = DataLoader(dataset, batch_size=config['batch_size'])
    y_predicted, y_true, loss = evaluate(model, loader, config)
    print(f'Prediction loss: {loss}')
    if save:
        np.savetxt('predicted.txt', expit(y_predicted.cpu().numpy()), fmt='%1.2f')
        np.savetxt('true.txt', y_true.cpu().numpy(), fmt='%1d')
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(y_predicted)
    predictions = (probs > config['threshold']).float().cpu().numpy()
    return predictions, y_true.cpu().numpy()


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


def classification(model, dataset, config):
    labels = config["label_names"]
    y_predicted, y_true = predict(model, dataset, config)
    print(classification_report(y_true, y_predicted, target_names=labels, digits=3), '\n')
    objects = make_objects(dataset.text, y_predicted, y_true, labels)
    return classification_report(y_true, y_predicted, target_names=labels, output_dict=True, zero_division=0), objects


def classify_main(config):
    beginning_t0 = time.time()
    set_seeds(42)
    test_path = config["data_file"]
    simcse_model = SimCSE(config['simcse_model'], device=config['device'])
    classifier_dataset = dataset.SimCSEMFTCDataset(data_file=test_path, sim_model=simcse_model, default_config=config, use_foundations=config["use_foundations"], label_names=config.get('label_names'))
    print("Device: ", config['device'])

    all_classifications = []
    f1_scores = defaultdict(list)
    for train_dataset, test_dataset in classifier_dataset.kfold(5):
        model = LinearCLS(input_size=config['input_size'], output_size=len(config['label_names']), dropout_rate=config["dropout"])
        model.to(config['device'])
        model = train(model, train_dataset, config)

        print('======== Statistics on test set ========')
        clf_report, obs = classification(model=model, dataset=test_dataset, config=config)
        all_classifications.extend(obs)

        for label in config["label_names"] + ['micro avg', 'macro avg', 'weighted avg']:
            f1_scores[label].append(clf_report[label]['f1-score'])

        print(f1_scores)

    with open(f'{config["simcse_model"].split("/")[1]}.json', 'w') as file:
        json.dump(all_classifications, file)

    print_results(f1_scores, config['label_names'], config["simcse_model"].split("/")[1])

    print(f'Total classification time: {format_time(time.time() - beginning_t0)}')


def classify_main_ablation(config):
    for i in range(len(config['simcse_model'])):
        beginning_t0 = time.time()
        set_seeds(42)
        test_path = config["data_file"]

        simcse_model = SimCSE(config['simcse_model'][i], device=config['device'])
        classifier_dataset = dataset.SimCSEMFTCDataset(data_file=test_path, sim_model=simcse_model, default_config=config, use_foundations=config["use_foundations"], label_names=config.get('label_names'))

        print("Device: ", config['device'])
        all_classifications = []
        f1_scores = defaultdict(list)
        for train_dataset, test_dataset in classifier_dataset.kfold(5):
            model = LinearCLS(input_size=config['input_size'], output_size=len(config['label_names']), dropout_rate=config["dropout"])
            model.to(config['device'])
            model = train(model, train_dataset, config)
            clf_report, obs = classification(model=model, dataset=test_dataset, config=config)
            all_classifications.extend(obs)

            for label in config["label_names"] + ['micro avg', 'macro avg', 'weighted avg']:
                f1_scores[label].append(clf_report[label]['f1-score'])

            print(clf_report)

        with open(f'{config["simcse_model"][i].split("/")[1]}.json', 'w') as file:
            json.dump(all_classifications, file)

        print_results(f1_scores, config['label_names'], config["simcse_model"][i].split("/")[1])

        print(f'Total classification time: {format_time(time.time() - beginning_t0)}')

if __name__ == "__main__":
    # classify_main(default_config)
    classify_main_ablation(ablation_config)
