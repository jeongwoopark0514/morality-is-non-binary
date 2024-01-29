import numpy as np
import pandas as pd
import random



def merge_virtue_vice(df: pd.DataFrame):
    df["virtue"] = df["care"] + df["fairness"] + df["loyalty"] + df["authority"] + df["purity"]
    df["vice"] = df["harm"] + df["cheating"] + df["betrayal"] + df["subversion"] + df["degradation"]

    df["virtue"] = df["virtue"].clip(upper=1)
    df["vice"] = df["vice"].clip(upper=1)

    virtue_df = df.query("virtue > 0 & vice == 0")
    vice_df = df.query("vice > 0 & virtue == 0")
    non_moral_df = df.query("`non-moral` == 1")
    full_df = pd.concat([virtue_df, vice_df, non_moral_df])
    full_df = full_df.sample(frac=1, random_state=42)
    arr = np.array_split(full_df, 2)
    return arr[0], arr[1]


def generate_binary_dataset(data_file: str):
    set_seeds(42)
    df = pd.read_csv(data_file)
    df1, df2 = merge_virtue_vice(df)
    output_df = pd.DataFrame(columns=['sent0', 'sent1', 'hard_neg'])
    output_df = generate_binary_supervised(df1, output_df, True)
    output_df = generate_binary_supervised(df2, output_df, False)

    output_df.reset_index(drop=True)
    output_df.to_csv(f"supervised_binary.csv", index=False)


def generate_binary_supervised(df: pd.DataFrame, output_df: pd.DataFrame, virtue: bool):
    df = df.reset_index(drop=True)
    non_moral_negative_df = df[df['non-moral'] == 1].sample(frac=1, random_state=42)

    # exclude non-morals
    df = df.drop(df[df["non-moral"] == 1].index).reset_index(drop=True)


    grouped = (df.groupby(["virtue", "vice"])
           .apply(lambda x: x.index.tolist())
           .reset_index(name='matching'))

    # virtue
    # Generate supervised sets
    non_moral_counter = 0

    if virtue:
        positives = grouped.query("virtue > 0")
        negatives = grouped.query("vice > 0")
    else:
        positives = grouped.query("vice > 0")
        negatives = grouped.query("virtue > 0")

    positive_index = 0
    instances = positives['matching'].tolist()[0]
    # Leave one item out if the number of matching pair is odd.
    if len(instances) % 2 == 1:
        max_pairs = len(instances) - 1
    else:
        max_pairs = len(instances)

    negatives_list = negatives['matching'].tolist()[0]
    for i in range(0, max_pairs, 2):
        sent0 = df.iloc[instances[i]]['processed']
        sent1 = df.iloc[instances[i+1]]['processed']

        if len(negatives_list) > 0:
            hard_neg = df.iloc[negatives_list.pop(0)]['processed']
        else:
            hard_neg = non_moral_negative_df.iloc[non_moral_counter]['processed']
            non_moral_counter += 1
        new_df = pd.DataFrame([sent0, sent1, hard_neg], index=["sent0", "sent1", "hard_neg"] )
        output_df = pd.concat([output_df, new_df.T], ignore_index=True)

        positive_index = i+1

    positives_list = positives["matching"].tolist()[0][positive_index+1:]
    negatives_list = negatives["matching"].tolist()[0]
    hard_neg_candidate_list = positives_list + negatives_list

    random.seed(42)
    print(len(positives_list), len(negatives_list))

    random.shuffle(hard_neg_candidate_list)

    for j in range(non_moral_counter, len(non_moral_negative_df)-1, 2):
        sent0 = non_moral_negative_df.iloc[j]['processed']
        non_moral_counter += 1
        sent1 = non_moral_negative_df.iloc[j+1]['processed']
        non_moral_counter += 1

        hard_neg = None

        if len(hard_neg_candidate_list) > 0:
            hard_neg = df.iloc[hard_neg_candidate_list.pop(0)]["processed"]

        if hard_neg is not None:
            new_df = pd.DataFrame([sent0, sent1, hard_neg], index=["sent0", "sent1", "hard_neg"] )
            output_df = pd.concat([output_df, new_df.T], ignore_index=True)
        else:
            break

    return output_df


def generate_3_way_test_set(data_file: str):
    set_seeds(42)

    df = pd.read_csv(data_file)

    df1, df2 = merge_virtue_vice(df)

    test_df = pd.concat([df1, df2]).reset_index(drop=True)

    output_df = test_df[["processed", "virtue", "vice", "non-moral"]]
    output_df.to_csv("binary_test.csv", index=False)


def main():
    set_seeds(42)
    df1 = pd.read_csv("processed/mftc/train/shuffled_merged_MFTC_train_add.csv")
    df2 = pd.read_csv("processed/mftc/test/shuffled_merged_MFTC_test_add.csv")
    full_df = pd.concat([df1, df2]).reset_index(drop=True)
    full_df.to_csv("processed/mftc/shuffled_merged_MFTC_all_add.csv", index=False)
    # df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # df.to_csv("processed/mftc/test/shuffled_merged_MFTC_train_add.csv", index=False)
    # generate_binary_dataset("processed/mftc/train/merged_MFTC_train_add.csv")
    # generate_3_way_test_set("processed/mftc/test/merged_MFTC_test_add.csv")

def set_seeds(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)

main()