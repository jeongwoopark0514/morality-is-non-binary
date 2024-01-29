import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split, KFold
from simcse import SimCSE
import torch

# The moral foundations dict with virtue as key and vice as value
MORAL_FOUNDATIONS = {
    'care': 'harm',
    'fairness': 'cheating',
    'loyalty': 'betrayal',
    'authority': 'subversion',
    'purity': 'degradation',
    'non-moral': 'non-moral'
}

class MFTCDataset(Dataset):
    def __init__(self, data_file=None, use_foundations=False, label_names=None, texts=None, labels=None,
                 max_size=None):
        self.text = texts
        self.labels = labels
        self.label_names = label_names

        if data_file is not None:
            if isinstance(data_file, str):
                self.data = pd.read_csv(data_file).dropna()
            else:
                self.data = data_file

            if max_size is None:
                max_size = self.data.shape[0]

            self.text = self.data['processed'].to_list()[:max_size]
            # self.ids = self.data['tweet_id'].to_numpy()[:max_size]

            if use_foundations:
                if self.label_names is None:
                    self.label_names = MORAL_FOUNDATIONS.keys()

                for l_name in self.label_names:
                    if l_name not in MORAL_FOUNDATIONS:
                        raise KeyError(f'Foundation {l_name} does not exist')

                num_rows = min(self.data.shape[0], max_size)
                num_cols = len(self.label_names)
                self.labels = np.empty([num_rows, num_cols], dtype=bool)
                for i, key in enumerate(self.label_names):
                    value = MORAL_FOUNDATIONS.get(key)
                    self.labels[:, i] = self.data[key].to_numpy()[:max_size] | self.data[value].to_numpy()[:max_size]
                self.labels = self.labels.astype(int)
            else:
                if self.label_names is None:
                    self.label_names = [x for x in self.data.columns if x != 'text']

                for l_name in self.label_names:
                    if l_name not in self.data.columns:
                        raise KeyError(f'Moral label {l_name} does not exist')

                self.labels = self.data[self.label_names].to_numpy()[:max_size]

    def __getitem__(self, index):
        return {
                'text': self.text[index],
                'labels': self.labels[index]}

    def __len__(self):
        return len(self.text)

class SimCSEMFTCDataset(MFTCDataset):

    def __init__(self, default_config: dict, sim_model: SimCSE, data_file=None, use_foundations=False, label_names=None, texts=None, labels=None,
                 max_size=35000):
        super().__init__(data_file, use_foundations, label_names, texts, labels, max_size)
        self.configs = default_config
        self.simcse_model = sim_model
        if data_file is None:
            self.encodings = self.simcse_model.encode(self.text, batch_size=default_config["batch_size"], max_length=default_config['max_length'])

        self.labels = torch.tensor(self.labels)

    def __getitem__(self, idx):
        return {'encodings': self.encodings[idx], 'labels': self.labels[idx]}

    def __len__(self):
        return len(self.labels)

    def kfold(self, k=10):
        kf = KFold(n_splits=k)

        texts = np.array(self.text)
        labels = self.labels

        for train_index, test_index in kf.split(X=self.text, y=self.labels):
            X_train, X_test = texts[train_index], texts[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            train_dataset = SimCSEMFTCDataset(self.configs, sim_model=self.simcse_model, data_file=None, texts=X_train.tolist(), labels=y_train,
                                        label_names=self.label_names)
            test_dataset = SimCSEMFTCDataset(self.configs, sim_model=self.simcse_model, data_file=None, texts=X_test.tolist(), labels=y_test,
                                       label_names=self.label_names)

            yield train_dataset, test_dataset

    def to(self, device):
        self.encodings = self.encodings.to(device)
        self.labels = self.labels.to(device)

class MFTCDistributionDataset(Dataset):
    def __init__(self, data_file=None, use_foundations=False, label_names=None, texts=None, labels=None,
                 max_size=None):
        self.text = texts
        self.labels = labels
        self.label_names = label_names

        if data_file is not None:
            if isinstance(data_file, str):
                self.data = pd.read_csv(data_file).dropna()
            else:
                self.data = data_file

            if max_size is None:
                max_size = self.data.shape[0]

            self.text = self.data['processed'].to_list()[:max_size]
            # self.ids = self.data['tweet_id'].to_numpy()[:max_size]

            if use_foundations:
                if self.label_names is None:
                    self.label_names = MORAL_FOUNDATIONS.keys()

                for l_name in self.label_names:
                    if l_name not in MORAL_FOUNDATIONS:
                        raise KeyError(f'Foundation {l_name} does not exist')

                num_rows = min(self.data.shape[0], max_size)
                num_cols = len(self.label_names)
                self.labels = np.empty([num_rows, num_cols], dtype=bool)
                for i, key in enumerate(self.label_names):
                    value = MORAL_FOUNDATIONS.get(key)
                    self.labels[:, i] = self.data[key].to_numpy()[:max_size] + self.data[value].to_numpy()[:max_size]
                self.labels = self.labels.astype(float)
            else:
                if self.label_names is None:
                    self.label_names = [x for x in self.data.columns if x != 'text']

                for l_name in self.label_names:
                    if l_name not in self.data.columns:
                        raise KeyError(f'Moral label {l_name} does not exist')

                self.labels = self.data[self.label_names].to_numpy()[:max_size]

    def __getitem__(self, index):
        return {
                'text': self.text[index],
                'labels': self.labels[index]}

    def __len__(self):
        return len(self.text)

class SimCSEDistrMFTCDataset(MFTCDistributionDataset):

    def __init__(self, default_config: dict, sim_model: SimCSE, data_file=None, use_foundations=False, label_names=None, texts=None, labels=None,
                 max_size=35000):
        super().__init__(data_file, use_foundations, label_names, texts, labels, max_size)
        self.configs = default_config
        self.simcse_model = sim_model
        if data_file is None:
            self.encodings = self.simcse_model.encode(self.text, batch_size=default_config["batch_size"], max_length=default_config['max_length'])

        self.labels = torch.tensor(self.labels)

    def __getitem__(self, idx):
        return {'encodings': self.encodings[idx], 'labels': self.labels[idx]}

    def __len__(self):
        return len(self.labels)

    def kfold(self, k=10):
        kf = KFold(n_splits=k)

        texts = np.array(self.text)
        labels = self.labels

        for train_index, test_index in kf.split(X=self.text, y=self.labels):
            X_train, X_test = texts[train_index], texts[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            train_dataset = SimCSEMFTCDataset(self.configs, sim_model=self.simcse_model, data_file=None, texts=X_train.tolist(), labels=y_train,
                                        label_names=self.label_names)
            test_dataset = SimCSEMFTCDataset(self.configs, sim_model=self.simcse_model, data_file=None, texts=X_test.tolist(), labels=y_test,
                                       label_names=self.label_names)

            yield train_dataset, test_dataset

    def to(self, device):
        self.encodings = self.encodings.to(device)
        self.labels = self.labels.to(device)
