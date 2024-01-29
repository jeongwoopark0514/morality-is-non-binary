import preprocess_mftc
import pandas as pd
import random
import numpy as np
import torch

moralValues = ["fairness", "non-moral", "purity", "degradation", "loyalty", "care", "cheating", "betrayal",
               "subversion", "authority", "harm"]
# MFTC
datasets = ["ALM", "Baltimore", "BLM", "Davidson", "Election", "MeToo", "Sandy"]

fract = 0.8


def generate_train_test_data():
    # Remove duplicated data for each dataset domain
    datasets = ["ALM", "Baltimore", "BLM", "Davidson", "Election", "MeToo", "Sandy"]
    for i in range(len(datasets)):
        preprocess_mftc.remove_dup_final(f"processed/mftc/{datasets[i]}.csv", datasets[i])

    # Split the merged dataset into train and test dataset.
    for i in range(len(datasets)):
        df_train, df_test = train_test_split_csv(f"processed/mftc/{datasets[i]}_add.csv", fract)
        df_train.to_csv(f"processed/mftc/train/fract80/{datasets[i]}_train.csv", index=False)
        df_test.to_csv(f"processed/mftc/test/fract80/{datasets[i]}_test.csv", index=False)

    # Then merge all domains into one.
    def mergeCorpuses(train_or_test: str):
        combined_csv = pd.concat([pd.read_csv(f'processed/mftc/{train_or_test}/fract80/' + f'{corpus}_{train_or_test}.csv') for corpus in datasets])
        # export all to csv
        combined_csv.to_csv(f'processed/mftc/{train_or_test}/fract80/merged_MFTC_{train_or_test}_80.csv', index=False, encoding='utf-8-sig')

    mergeCorpuses("train")
    mergeCorpuses("test")

    # Remove duplicates again to prevent duplicates across domains
    preprocess_mftc.remove_dup_final(f"processed/mftc/train/fract80/merged_MFTC_train_80.csv", "all")
    # Remove duplicates again to prevent duplicates across domains
    preprocess_mftc.remove_dup_final(f"processed/mftc/test/fract80/merged_MFTC_test_80.csv", "all")


def train_test_split_csv(path_name: str, ratio: float):
    df = pd.read_csv(path_name)
    df_train = df.groupby(moralValues, group_keys=False).apply(lambda x: x.sample(frac=ratio, random_state=42))
    df_test = df[~df.index.isin(df_train.index)]

    return df_train, df_test


def set_seeds(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

if __name__ == '__main__':
    set_seeds()
    train, test = train_test_split_csv("processed/mftc/MFTC.csv", 0.9)
