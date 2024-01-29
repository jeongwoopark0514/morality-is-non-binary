import random

import numpy as np
import pandas as pd
from itertools import chain, combinations
from utils import generate_foundation_columns


foundation_cols = ['careF', 'fairnessF', 'loyaltyF', 'authorityF', 'purityF', 'non-moral']
duality_cols = ['fairness', 'non-moral', 'purity', 'degradation', 'loyalty', 'care', 'cheating', 'betrayal',
               'subversion', 'authority', 'harm']

DUALITY_INDEX = {
    'care' : 0,
    'harm' : 1,
    'fairness' : 2,
    'cheating' : 3,
    'loyalty' : 4,
    'betrayal' : 5,
    'authority' : 6,
    'subversion' : 7,
    'purity' : 8,
    'degradation' : 9
}

VICE_VIRTUE = {
    'care' : 'harm',
    'harm' : 'care',
    'fairness' : 'cheating',
    'cheating' : 'fairness',
    'loyalty' : 'betrayal',
    'betrayal' : 'loyalty',
    'authority' : 'subversion',
    'subversion' : 'authority',
    'purity' : 'degradation',
    'degradation' : 'purity'
}


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def generate_sup_duality_half(file_path: str):
    df = pd.read_csv(file_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    arr = np.array_split(df, 2)
    df1 = arr[0]
    df2 = arr[1]

    df1.to_csv(file_path.split('.')[0] + "_1.csv", index=False)
    df2.to_csv(file_path.split('.')[0] + "_2.csv", index=False)

    generate_sup_duality_outside_foundation(file_path.split('.')[0] + "_1.csv")
    generate_sup_duality_within_foundation(file_path.split('.')[0] + "_2.csv")

    # generate_sup_duality_outside_foundation()


def generate_sup_duality_outside_foundation(file_path: str):
    df = pd.read_csv(file_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    output_df = pd.DataFrame(columns=['sent0', 'sent1', 'hard_neg'])

    # Create foundation columns
    non_moral_negative = df[df['non-moral'] == 1].sample(frac=1, random_state=42)

    # exclude non-morals
    df = df.drop(df[df["non-moral"] == 1].index).reset_index(drop=True)

    df_duality = (df.groupby(duality_cols)
           .apply(lambda x: x.index.tolist())
           .reset_index(name='matching'))


    # Make a column for the count of foundations
    df_duality['numD'] = df_duality["fairness"] + df_duality["non-moral"] + df_duality["purity"] + df_duality["degradation"] \
                        + df_duality["loyalty"] + df_duality["care"] + df_duality["cheating"] + df_duality["betrayal"] \
                        + df_duality["subversion"] + df_duality["authority"] + df_duality["harm"]

    # Make a column that has the number of matching pairs
    # duality_cols = ['fairness', 'non-moral', 'purity', 'degradation', 'loyalty', 'care', 'cheating', 'betrayal',
                    # 'subversion', 'authority', 'harm']
    df_duality['num_matching'] = df_duality['matching'].apply(len)

    df_duality = df_duality.sort_values(by=['numD', 'num_matching'], ascending=[False, True]).reset_index()

    # Add an empty column
    df_duality["neg_morals"] = np.empty((len(df_duality), 0)).tolist()

    # Find opposing pairs
    for index, row in df_duality.iterrows():
        df_duality = get_reverse_candidates_duality_another_foundation(df_duality, index)

    df_duality['num_matching'] = df_duality['matching'].apply(len)
    df_duality['num_neg_morals'] = df_duality['neg_morals'].apply(len)


    # Generate supervised sets
    non_moral_counter = 0
    for index, row in df_duality.iterrows():
        instances = row['matching']

        # Leave one item out if the number of matching pair is odd.
        if row['num_matching'] % 2 == 1:
            max_pairs = row['num_matching'] - 1
        else:
            max_pairs = row['num_matching']

        for i in range(0, max_pairs, 2):
            sent0 = df.iloc[instances[i]]['processed']
            sent1 = df.iloc[instances[i+1]]['processed']

            if len(row['neg_morals']) > 0:
                hard_neg = df.iloc[row['neg_morals'].pop(0)]['processed']
            else:
                # print(non_moral_counter, non_moral_negative.shape)
                hard_neg = non_moral_negative.iloc[non_moral_counter]['processed']
                non_moral_counter += 1
            new_df = pd.DataFrame([sent0, sent1, hard_neg], index=["sent0", "sent1", "hard_neg"] )
            output_df = pd.concat([output_df, new_df.T], ignore_index=True)

    counter=0

    for j in range(non_moral_counter, len(non_moral_negative)-1, 2):
        sent0 = non_moral_negative.iloc[j]['processed']
        non_moral_counter += 1
        sent1 = non_moral_negative.iloc[j+1]['processed']
        non_moral_counter += 1

        hard_neg = None
        for index, row in df_duality.iterrows():
            if len(row['neg_morals']) > 0:
                hard_neg = df.iloc[row['neg_morals'].pop(0)]['processed']
                break
        if hard_neg is not None:
            new_df = pd.DataFrame([sent0, sent1, hard_neg], index=["sent0", "sent1", "hard_neg"] )
            output_df = pd.concat([output_df, new_df.T], ignore_index=True)
            counter += 1
    print("Non-moral ", counter)
    output_df.reset_index(drop=True)
    output_df.to_csv(file_path.split('.')[0] + "_test1.csv", index=False)




def get_reverse_candidates_duality_another_foundation(df: pd.DataFrame, idx: int):
    care = df.iloc[idx]['care']
    harm = df.iloc[idx]['harm']
    fairness = df.iloc[idx]['fairness']
    cheating = df.iloc[idx]['cheating']
    loyalty = df.iloc[idx]['loyalty']
    betrayal = df.iloc[idx]['betrayal']
    authority = df.iloc[idx]['authority']
    subversion = df.iloc[idx]['subversion']
    purity = df.iloc[idx]['purity']
    degradation = df.iloc[idx]['degradation']

    currentVals = np.array([care, harm, fairness, cheating, loyalty, betrayal, authority, subversion, purity, degradation])

    unprioritised = []

    for key, value in DUALITY_INDEX.items():
        if currentVals[value] == 1 :
            unprioritised.append(DUALITY_INDEX[VICE_VIRTUE[key]])

    unprioritised = set(np.array(unprioritised))
    opposite_candiates = set(np.where(currentVals == 0)[0])
    test = opposite_candiates - unprioritised
    opposites = test
    subsets = list(powerset(opposites))

    # [0, 3, 4] -> [1, 0, 0, 1, 1]
    stop = False

    if len(df.iloc[idx]['matching']) == 0:
        return df
    for ex in reversed(subsets):
        i_list = np.zeros(10)
        for i in ex:
            i_list[i] = 1

        for index, row in df.iterrows():
            if int(index) > idx and len(df.iloc[index]['matching']) > 0:
                row_arr = [row['care'], row['harm'], row['fairness'], row['cheating'], row['loyalty'], row['betrayal'], row['authority'], row['subversion'], row['purity'], row['degradation']]

                if (np.array(row_arr) == np.array(i_list)).all():
                    matching_size = len(df.iloc[index]['matching'])
                    for i in range(matching_size):
                        df.iloc[idx]['neg_morals'] = df.iloc[idx]['neg_morals'].append(df.iloc[index]['matching'][0])
                        df.iloc[index]['matching'].pop(0)

                        if abs(len(df.iloc[idx]['matching']) - len(df.iloc[idx]['neg_morals']) * 2) <= 1:
                            stop = True
                            return df
            if stop:
                return df
        if stop:
            return df

    return df


def generate_sup_duality_within_foundation(file_path: str):
    df = pd.read_csv(file_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    output_df = pd.DataFrame(columns=['sent0', 'sent1', 'hard_neg'])

    # Create foundation columns
    non_moral_negative = df[df['non-moral'] == 1].sample(frac=1, random_state=42)

    # exclude non-morals
    df = df.drop(df[df["non-moral"] == 1].index).reset_index(drop=True)

    df_duality = (df.groupby(duality_cols)
           .apply(lambda x: x.index.tolist())
           .reset_index(name='matching'))

    # Make a column for the count of foundations
    df_duality['numD'] = df_duality["fairness"] + df_duality["non-moral"] + df_duality["purity"] + df_duality["degradation"] \
                        + df_duality["loyalty"] + df_duality["care"] + df_duality["cheating"] + df_duality["betrayal"] \
                        + df_duality["subversion"] + df_duality["authority"] + df_duality["harm"]

    # Make a column that has the number of matching pairs
    # duality_cols = ['fairness', 'non-moral', 'purity', 'degradation', 'loyalty', 'care', 'cheating', 'betrayal',
                    # 'subversion', 'authority', 'harm']
    df_duality['num_matching'] = df_duality['matching'].apply(len)

    df_duality = df_duality.sort_values(by=['numD', 'num_matching'], ascending=[False, True]).reset_index()

    # Add an empty column
    df_duality["neg_morals"] = np.empty((len(df_duality), 0)).tolist()

    # Find opposing pairs
    for index, row in df_duality.iterrows():
        df_duality = get_reverse_candidates_duality_within_foundation(df_duality, index)

    df_duality['num_matching'] = df_duality['matching'].apply(len)
    df_duality['num_neg_morals'] = df_duality['neg_morals'].apply(len)

    # Generate supervised sets
    non_moral_counter = 0
    for index, row in df_duality.iterrows():
        instances = row['matching']

        # Leave one item out if the number of matching pair is odd.
        if row['num_matching'] % 2 == 1:
            max_pairs = row['num_matching'] - 1
        else:
            max_pairs = row['num_matching']
        for i in range(0, max_pairs, 2):
            sent0 = df.iloc[instances[i]]['processed']
            sent1 = df.iloc[instances[i+1]]['processed']
            if len(row['neg_morals']) > 0:
                hard_neg = df.iloc[row['neg_morals'].pop(0)]['processed']
            else:
                # print(non_moral_counter, non_moral_negative.shape)
                hard_neg = non_moral_negative.iloc[non_moral_counter]['processed']
                non_moral_counter += 1
            new_df = pd.DataFrame([sent0, sent1, hard_neg], index=["sent0", "sent1", "hard_neg"] )
            output_df = pd.concat([output_df, new_df.T], ignore_index=True)

    counter=0

    for j in range(non_moral_counter, len(non_moral_negative)-1, 2):
        sent0 = non_moral_negative.iloc[j]['processed']
        non_moral_counter += 1
        sent1 = non_moral_negative.iloc[j+1]['processed']
        non_moral_counter += 1

        hard_neg = None
        for index, row in df_duality.iterrows():
            if len(row['neg_morals']) > 0:
                hard_neg = df.iloc[row['neg_morals'].pop(0)]['processed']
                break
        if hard_neg is not None:
            new_df = pd.DataFrame([sent0, sent1, hard_neg], index=["sent0", "sent1", "hard_neg"] )
            output_df = pd.concat([output_df, new_df.T], ignore_index=True)
            counter += 1
    print("Non-moral ", counter)
    output_df.reset_index(drop=True)
    output_df.to_csv(file_path.split('.')[0] + "test2.csv", index=False)



def get_reverse_candidates_duality_within_foundation(df: pd.DataFrame, idx: int):
    care = df.iloc[idx]['care']
    harm = df.iloc[idx]['harm']
    fairness = df.iloc[idx]['fairness']
    cheating = df.iloc[idx]['cheating']
    loyalty = df.iloc[idx]['loyalty']
    betrayal = df.iloc[idx]['betrayal']
    authority = df.iloc[idx]['authority']
    subversion = df.iloc[idx]['subversion']
    purity = df.iloc[idx]['purity']
    degradation = df.iloc[idx]['degradation']

    currentVals = np.array([care, harm, fairness, cheating, loyalty, betrayal, authority, subversion, purity, degradation])

    prioritised = []

    for key, value in DUALITY_INDEX.items():
        if currentVals[value] == 1 :
            prioritised.append(DUALITY_INDEX[VICE_VIRTUE[key]])

    prioritised = set(np.array(prioritised))
    subsets = list(powerset(prioritised))

    # [0, 3, 4] -> [1, 0, 0, 1, 1]
    stop = False


    if len(df.iloc[idx]['matching']) == 0:
        return df
    for ex in reversed(subsets):
        i_list = np.zeros(10)
        for i in ex:
            i_list[i] = 1

        for index, row in df.iterrows():
            if int(index) > idx and len(df.iloc[index]['matching']) > 0:
                row_arr = [row['care'], row['harm'], row['fairness'], row['cheating'], row['loyalty'], row['betrayal'], row['authority'], row['subversion'], row['purity'], row['degradation']]

                if (np.array(row_arr) == np.array(i_list)).all():
                    matching_size = len(df.iloc[index]['matching'])
                    for i in range(matching_size):
                        df.iloc[idx]['neg_morals'] = df.iloc[idx]['neg_morals'].append(df.iloc[index]['matching'][0])
                        df.iloc[index]['matching'].pop(0)
                        # print(len(df.iloc[idx]['matching']), len(df.iloc[idx]['neg_morals']))
                        if abs(len(df.iloc[idx]['matching']) - len(df.iloc[idx]['neg_morals']) * 2) <= 1:
                            stop = True
                            return df
            if stop:
                return df
        if stop:
            return df

    return df

def generate_duality(file_path: str):
    df = pd.read_csv(file_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    output_df = pd.DataFrame(columns=['sent0', 'sent1', 'hard_neg'])

    # Create foundation columns
    non_moral_negative = df[df['non-moral'] == 1].sample(frac=1, random_state=42)

    # exclude non-morals
    df = df.drop(df[df["non-moral"] == 1].index).reset_index(drop=True)

    df_duality = (df.groupby(duality_cols)
           .apply(lambda x: x.index.tolist())
           .reset_index(name='matching'))


    # Make a column for the count of foundations
    df_duality['numD'] = df_duality["fairness"] + df_duality["non-moral"] + df_duality["purity"] + df_duality["degradation"] \
                        + df_duality["loyalty"] + df_duality["care"] + df_duality["cheating"] + df_duality["betrayal"] \
                        + df_duality["subversion"] + df_duality["authority"] + df_duality["harm"]

    # Make a column that has the number of matching pairs
    # duality_cols = ['fairness', 'non-moral', 'purity', 'degradation', 'loyalty', 'care', 'cheating', 'betrayal',
                    # 'subversion', 'authority', 'harm']
    df_duality['num_matching'] = df_duality['matching'].apply(len)

    df_duality = df_duality.sort_values(by=['numD', 'num_matching'], ascending=[False, True]).reset_index()

    # Add an empty column
    df_duality["neg_morals"] = np.empty((len(df_duality), 0)).tolist()

    # Find opposing pairs
    for index, row in df_duality.iterrows():
        df_duality = get_reverse_candidates(df_duality, index)

    df_duality['num_matching'] = df_duality['matching'].apply(len)
    df_duality['num_neg_morals'] = df_duality['neg_morals'].apply(len)

    # Generate supervised sets
    non_moral_counter = 0
    for index, row in df_duality.iterrows():
        instances = row['matching']

        # Leave one item out if the number of matching pair is odd.
        if row['num_matching'] % 2 == 1:
            max_pairs = row['num_matching'] - 1
        else:
            max_pairs = row['num_matching']

        for i in range(0, max_pairs, 2):
            sent0 = df.iloc[instances[i]]['processed']
            sent1 = df.iloc[instances[i+1]]['processed']

            if len(row['neg_morals']) > 0:
                hard_neg = df.iloc[row['neg_morals'].pop(0)]['processed']
            else:
                # print(non_moral_counter, non_moral_negative.shape)
                hard_neg = non_moral_negative.iloc[non_moral_counter]['processed']
                non_moral_counter += 1
            new_df = pd.DataFrame([sent0, sent1, hard_neg], index=["sent0", "sent1", "hard_neg"] )
            output_df = pd.concat([output_df, new_df.T], ignore_index=True)
    counter=0

    for j in range(non_moral_counter, len(non_moral_negative), 2):
        sent0 = non_moral_negative.iloc[j]['processed']
        non_moral_counter += 1
        sent1 = non_moral_negative.iloc[j+1]['processed']
        non_moral_counter += 1

        hard_neg = None
        for index, row in df_duality.iterrows():
            if len(row['neg_morals']) > 0:
                hard_neg = df.iloc[row['neg_morals'].pop(0)]['processed']
                break
        if hard_neg is not None:
            new_df = pd.DataFrame([sent0, sent1, hard_neg], index=["sent0", "sent1", "hard_neg"] )
            output_df = pd.concat([output_df, new_df.T], ignore_index=True)
            counter += 1
    print("Non-moral ", counter)
    output_df.reset_index(drop=True)
    output_df.to_csv(file_path.split('.')[0] + "_with_non_moral.csv", index=False)


def get_reverse_candidates(df: pd.DataFrame, idx: int):
    care = df.iloc[idx]['care']
    harm = df.iloc[idx]['harm']
    fairness = df.iloc[idx]['fairness']
    cheating = df.iloc[idx]['cheating']
    loyalty = df.iloc[idx]['loyalty']
    betrayal = df.iloc[idx]['betrayal']
    authority = df.iloc[idx]['authority']
    subversion = df.iloc[idx]['subversion']
    purity = df.iloc[idx]['purity']
    degradation = df.iloc[idx]['degradation']

    currentVals = np.array([care, harm, fairness, cheating, loyalty, betrayal, authority, subversion, purity, degradation])

    prioritised = []

    for key, value in DUALITY_INDEX.items():
        if currentVals[value] != 1:
            prioritised.append(value)

    prioritised = set(np.array(prioritised))
    subsets = list(powerset(prioritised))

    # [0, 3, 4] -> [1, 0, 0, 1, 1]
    stop = False

    if len(df.iloc[idx]['matching']) == 0:
        return df
    for ex in reversed(subsets):
        i_list = np.zeros(10)
        for i in ex:
            i_list[i] = 1

        for index, row in df.iterrows():
            if int(index) > idx and len(df.iloc[index]['matching']) > 0:
                row_arr = [row['care'], row['harm'], row['fairness'], row['cheating'], row['loyalty'], row['betrayal'], row['authority'], row['subversion'], row['purity'], row['degradation']]

                if (np.array(row_arr) == np.array(i_list)).all():
                    matching_size = len(df.iloc[index]['matching'])
                    for i in range(matching_size):
                        df.iloc[idx]['neg_morals'] = df.iloc[idx]['neg_morals'].append(df.iloc[index]['matching'][0])
                        df.iloc[index]['matching'].pop(0)
                        # print(len(df.iloc[idx]['matching']), len(df.iloc[idx]['neg_morals']))
                        if abs(len(df.iloc[idx]['matching']) - len(df.iloc[idx]['neg_morals']) * 2) <= 1:
                            stop = True
                            return df
            if stop:
                return df
        if stop:
            return df

    return df


def main():

    # for i in range(1,6):
    #     generate_sup_duality_half(f"processed/mftc/5fold/train_set{i}.csv")
    # for i in range(1, 6):
    #     combined_csv = pd.concat([pd.read_csv(
    #         f'processed/mftc/5fold/train_set{i}_1_outside_with_non_moral.csv'),
    #                               pd.read_csv(
    #                                   f'processed/mftc/5fold/train_set{i}_2_within_with_non_moral.csv')])
    # # export all to csv
    # combined_csv.to_csv(f'processed/mftc/5fold/MFTC_train_set{i}_supervised.csv', index=False, encoding='utf-8-sig')

    generate_sup_duality_half(f"processed/mftc/test/merged_MFTC_test_add.csv")
    combined_csv = pd.concat([pd.read_csv(
        f'processed/mftc/test/merged_MFTC_test_add_1_test1.csv'),
                              pd.read_csv(
                                  f'processed/mftc/test/merged_MFTC_test_add_2test2.csv')])
    # export all to csv
    combined_csv.to_csv(f'processed/mftc/test/align_uniform_test.csv', index=False, encoding='utf-8-sig')





if __name__ == "__main__":
    main()