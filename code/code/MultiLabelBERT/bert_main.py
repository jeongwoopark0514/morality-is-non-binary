import json
from collections import defaultdict
import time
from transformers import AutoModelForSequenceClassification

from dataset import BertDataset
from multiLabelBERT import MultiLabelBertBase
from main_utils import set_seeds, combine_datasets, classification, print_results, get_dataset

from torch.nn import BCEWithLogitsLoss
from transformers import AutoTokenizer
from transformers import AdamW

moral_labels = ['care', 'harm',
                'fairness', 'cheating',
                'loyalty', 'betrayal',
                'authority', 'subversion',
                'purity', 'degradation',
                'non-moral']

moral_foundations = ['care', 'fairness', 'loyalty', 'authority', 'purity', 'non-moral']


def save_bert_train_model(data_file: str, save_model_name: str, use_foundations=False):
    """
    Train BertForSequenceClassification and save the model and its tokenizer.
    """
    model_name = "bert-large-uncased"

    # Model config
    MODEL_CONFIG = {
        "label_names": moral_foundations if use_foundations else moral_labels,
        "epochs": 10,
        "loss_fct": BCEWithLogitsLoss(),
        "threshold": 0,
        "batch_size": 16,
        "optim": AdamW,
        "name": model_name,
        "dropout": 0.1,
        "learning_rate": 2e-5
    }

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    set_seeds(seed_val)

    BertDataset.tokenizer = AutoTokenizer.from_pretrained(model_name)

    bert_dataset = BertDataset(data_file=data_file, use_foundations=use_foundations,
                               label_names=MODEL_CONFIG.get('label_names'), max_size=35000)

    if use_foundations:
        print('Using the foundations labels')
    else:
        print('Using the moral values labels')


    bert = MultiLabelBertBase(config=MODEL_CONFIG)

    train_dataset = bert_dataset.kfold_double_cross()
    train_start_time = time.time()
    bert.train(train_dataset, None, validation=False)
    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    bert.model.save_pretrained(save_model_name)
    BertDataset.tokenizer.save_pretrained(save_model_name)

    print(f'Time spent training {train_time}')

    return


def evaluate_bert_epoch10(data_file: str, saved_model: str, clause: str, do_kfold=True, kfold_num=5, use_foundations=True):
    model_name = saved_model

    # Model config
    MODEL_CONFIG = {
        "label_names": moral_foundations if use_foundations else moral_labels,
        "epochs": 10,
        "loss_fct": BCEWithLogitsLoss(),
        "threshold": 0,
        "batch_size": 16,
        "optim": AdamW,
        "name": model_name,
        "dropout": 0.1,
        "learning_rate": 2e-5
    }

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    set_seeds(seed_val)

    BertDataset.tokenizer = AutoTokenizer.from_pretrained(model_name)

    bert_dataset = BertDataset(data_file=data_file, use_foundations=use_foundations,
                               label_names=MODEL_CONFIG.get('label_names'), max_size=35000)

    if use_foundations:
        print('Using the foundations labels')
    else:
        print('Using the moral values labels')

    if do_kfold:
        f1_scores = defaultdict(list)

        durations = []
        all_classifications = []

        for index, (train_dataset, test_dataset) in enumerate(bert_dataset.kfold(kfold_num)):
            print('======== Statistics on test set ========')
            start_time = time.time()
            model = AutoModelForSequenceClassification.from_pretrained(model_name, return_dict=True)
            bert = MultiLabelBertBase(config=MODEL_CONFIG, model=model)
            bert.train(train_dataset, test_dataset, validation=False)
            clf_report, clf_obj = classification(bert, test_dataset, labels=MODEL_CONFIG['label_names'])
            end_time = time.time()
            durations.append(end_time - start_time)
            all_classifications.extend(clf_obj)

            for label in MODEL_CONFIG["label_names"] + ['micro avg', 'macro avg', 'weighted avg']:
                if label in clf_report:
                    f1_scores[label].append(clf_report[label]['f1-score'])


        print_results(f1_scores, MODEL_CONFIG['label_names'], f"result-multi-label-bert-{clause}-{kfold_num}fold-epoch10.txt")

        print(f'Average time spent classifying {sum(durations) / len(durations)}')
        with open(f'multi-label-bert-{clause}-{kfold_num}fold-epoch10.json', 'w') as file:
            json.dump(all_classifications, file)

        return
    else:
        f1_scores = defaultdict(list)

        durations = []
        all_classifications = []
        train_dataset, test_dataset = bert_dataset.kfold2()
        print("Train_dataset size: ", len(train_dataset))
        print("Test dataset size: ", len(test_dataset))

        bert = MultiLabelBertBase(config=MODEL_CONFIG)
        start_time = time.time()
        bert.train(train_dataset, test_dataset, validation=False)
        end_time = time.time()
        print('======== Statistics on test set ========')
        clf_report, clf_obj = classification(bert, test_dataset, labels=MODEL_CONFIG['label_names'])

        durations.append(end_time - start_time)
        all_classifications.extend(clf_obj)

        for label in MODEL_CONFIG["label_names"] + ['micro avg', 'macro avg', 'weighted avg']:
            if label in clf_report:
                f1_scores[label].append(clf_report[label]['f1-score'])

        bert.save_config()
        # print_results(f1_scores, MODEL_CONFIG['label_names'], "result-multi-label-bert.txt")
        print(f'Average time spent training {sum(durations) / len(durations)}')

        with open(f'multi-label-bert-{clause}-epoch10.json', 'w') as file:
            json.dump(all_classifications, file)

        return


def evaluate_bert_epoch3(data_file: str, clause: str, do_kfold=True, kfold_num=5, use_foundations=True):
    model_name = 'bert-large-uncased'

    # Model config
    MODEL_CONFIG = {
        "label_names": moral_foundations if use_foundations else moral_labels,
        "epochs": 3,
        "loss_fct": BCEWithLogitsLoss(),
        "threshold": 0,
        "batch_size": 16,
        "optim": AdamW,
        "name": model_name,
        "dropout": 0.1,
        "learning_rate": 2e-5
    }

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    set_seeds(seed_val)

    BertDataset.tokenizer = AutoTokenizer.from_pretrained(model_name)

    bert_dataset = BertDataset(data_file=data_file, use_foundations=use_foundations,
                               label_names=MODEL_CONFIG.get('label_names'), max_size=35000)

    if use_foundations:
        print('Using the foundations labels')
    else:
        print('Using the moral values labels')

    if do_kfold:
        f1_scores = defaultdict(list)

        durations = []
        all_classifications = []
        for train_dataset, test_dataset in bert_dataset.kfold(kfold_num):
            train_label_count = {
                "care": 0,
                "harm": 0,
                "fairness": 0,
                "cheating": 0,
                "loyalty": 0,
                "betrayal": 0,
                "authority": 0,
                "subversion": 0,
                "purity": 0,
                "degradation": 0,
                "non-moral": 0
            }
            print("Train_dataset size: ", len(train_dataset))
            print("Test dataset size: ", len(test_dataset))
            for i, labels in enumerate(train_dataset.labels):
                predicted_labels = [moral_labels[j] for j, x in enumerate(labels) if x == 1]

                for pred in predicted_labels:
                    train_label_count[pred] += 1

            print("Training set label distribution: ", train_label_count)
            bert = MultiLabelBertBase(config=MODEL_CONFIG)
            start_time = time.time()
            bert.train(train_dataset, test_dataset, validation=False)
            end_time = time.time()
            print('======== Statistics on test set ========')
            clf_report, clf_obj = classification(bert, test_dataset, labels=MODEL_CONFIG['label_names'])

            durations.append(end_time - start_time)
            all_classifications.extend(clf_obj)

            for label in MODEL_CONFIG["label_names"] + ['micro avg', 'macro avg', 'weighted avg']:
                if label in clf_report:
                    f1_scores[label].append(clf_report[label]['f1-score'])

            bert.save_config()

        print_results(f1_scores, MODEL_CONFIG['label_names'], f"result-multi-label-bert-{clause}-{kfold_num}fold-epoch3.txt")
        print(f'Average time spent training {sum(durations) / len(durations)}')

        with open(f'multi-label-bert-{clause}-{kfold_num}fold-epoch3.json', 'w') as file:
            json.dump(all_classifications, file)

        return
    else:
        f1_scores = defaultdict(list)

        durations = []
        all_classifications = []
        train_dataset, test_dataset = bert_dataset.kfold2()
        print("Train_dataset size: ", len(train_dataset))
        print("Test dataset size: ", len(test_dataset))

        bert = MultiLabelBertBase(config=MODEL_CONFIG)
        start_time = time.time()
        bert.train(train_dataset, test_dataset, validation=False)
        end_time = time.time()
        print('======== Statistics on test set ========')
        clf_report, clf_obj = classification(bert, test_dataset, labels=MODEL_CONFIG['label_names'])

        durations.append(end_time - start_time)
        all_classifications.extend(clf_obj)

        for label in MODEL_CONFIG["label_names"] + ['micro avg', 'macro avg', 'weighted avg']:
            if label in clf_report:
                f1_scores[label].append(clf_report[label]['f1-score'])


        # print_results(f1_scores, MODEL_CONFIG['label_names'], "result-multi-label-bert.txt")
        print(f'Average time spent training {sum(durations) / len(durations)}')

        with open(f'multi-label-bert-{clause}-epoch3.json', 'w') as file:
            json.dump(all_classifications, file)

        return

if __name__ == "__main__":
    # BERT ALL
    save_bert_train_model("data/MFTC_all_add.csv", './saved_model/multi-label-bert-trained', False)
    evaluate_bert_epoch10("data/merged_MFTC_test_add.csv", './saved_model/multi-label-bert-trained', 'weights', do_kfold=True, kfold_num=5, use_foundations=False)

    # BERT test-only
    evaluate_bert_epoch10("data/merged_MFTC_test_add.csv", "bert-large-uncased", "test_only", do_kfold=True, use_foundations=False)


