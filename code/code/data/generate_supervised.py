import numpy as np
import pandas as pd
from itertools import chain, combinations
from utils import generate_foundation_columns


foundation_cols = ['careF', 'fairnessF', 'loyaltyF', 'authorityF', 'purityF', 'non-moral']
duality_cols = ['fairness', 'non-moral', 'purity', 'degradation', 'loyalty', 'care', 'cheating', 'betrayal',
               'subversion', 'authority', 'harm']

foundation_index_pair = {
    0: 'careF',
    1: 'fairnessF',
    2: 'loyaltyF',
    3: 'authorityF',
    4: 'purityF'
}

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def generate_sup_foundation(file_path: str):
    df = pd.read_csv(file_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    output_df = pd.DataFrame(columns=['sent0', 'sent1', 'hard_neg'])

    # Create foundation columns
    non_moral_negative = df[df['non-moral'] == 1].sample(frac=1, random_state=42)

    # exclude non-morals
    df = df.drop(df[df["non-moral"] == 1].index).reset_index(drop=True)
    df_foundation = generate_foundation_columns(df)
    df_foundation = df_foundation.drop(
        columns=["care", "harm", "fairness", "cheating", "loyalty", "betrayal", "authority", "subversion", "purity",
                 "degradation"])

    df_foundation = (df_foundation.groupby(foundation_cols)
           .apply(lambda x: x.index.tolist())
           .reset_index(name='matching'))

    # Make a column for the count of foundations
    df_foundation['numF'] = df_foundation['careF'] + df_foundation['fairnessF'] + df_foundation['loyaltyF'] \
                             + df_foundation['authorityF'] + df_foundation['purityF']

    # Make a column that has the number of matching pairs
    df_foundation['num_matching'] = df_foundation['matching'].apply(len)

    df_foundation = df_foundation.sort_values(by=['numF', 'num_matching'], ascending=[False, True]).reset_index()

    # Add an empty column
    df_foundation["neg_morals"] = np.empty((len(df_foundation), 0)).tolist()

    # Create a for_non_moral array for non-morals
    for_non_morals = []

    # Find opposing pairs
    for index, row in df_foundation.iterrows():
        df_foundation = get_reverse_candidates_foundation(df_foundation, index)

    df_foundation['num_matching'] = df_foundation['matching'].apply(len)
    df_foundation['num_neg_morals'] = df_foundation['neg_morals'].apply(len)

    # Generate supervised sets
    non_moral_counter = 0
    for index, row in df_foundation.iterrows():
        instances = row['matching']

        # Leave one item out if the number of matching pair is odd.
        if row['num_matching'] % 2 == 1:
            max_pairs = row['num_matching'] - 1
        else:
            max_pairs = row['num_matching']

        for i in range(0, max_pairs, 2):
            sent0 = df.iloc[instances[i]]['processed']
            sent1 = df.iloc[instances[i+1]]['processed']
            if sent0 == sent1:
                continue
            if len(row['neg_morals']) > 0:
                hard_neg = df.iloc[row['neg_morals'].pop(0)]['processed']
            else:
                # print(non_moral_counter, non_moral_negative.shape)
                hard_neg = non_moral_negative.iloc[non_moral_counter]['processed']
                non_moral_counter += 1
            new_df = pd.DataFrame([sent0, sent1, hard_neg], index=["sent0", "sent1", "hard_neg"] )
            output_df = pd.concat([output_df, new_df.T], ignore_index=True)

    counter = 0
    for j in range(non_moral_counter, len(non_moral_negative)-1, 2):
        sent0 = non_moral_negative.iloc[j]['processed']
        non_moral_counter += 1
        sent1 = non_moral_negative.iloc[j+1]['processed']
        non_moral_counter += 1

        hard_neg = None
        for index, row in df_foundation.iterrows():
            if len(row['neg_morals']) > 0:
                hard_neg = df.iloc[row['neg_morals'].pop(0)]['processed']
                break
        if hard_neg is not None:
            new_df = pd.DataFrame([sent0, sent1, hard_neg], index=["sent0", "sent1", "hard_neg"] )
            output_df = pd.concat([output_df, new_df.T], ignore_index=True)
            counter += 1
    print("Non-moral ", counter)
    output_df.reset_index(drop=True)
    output_df.to_csv(file_path.split('.')[0] + "_foundation.csv", index=False)


def get_reverse_candidates_foundation(df: pd.DataFrame, idx: int):
    careVal = df.iloc[idx]['careF']
    fairnessVal = df.iloc[idx]['fairnessF']
    loyaltyVal = df.iloc[idx]['loyaltyF']
    authorityVal = df.iloc[idx]['authorityF']
    purityVal = df.iloc[idx]['purityF']

    currentVals = np.array([careVal, fairnessVal, loyaltyVal, authorityVal, purityVal])
    opposites = set(np.where(currentVals == 0)[0])
    subsets = list(powerset(opposites))
    # print(subsets, type(subsets))

    # [0, 3, 4] -> [1, 0, 0, 1, 1]
    stop = False

    if len(df.iloc[idx]['matching']) == 0:
        return df
    for ex in reversed(subsets):
        i_list = np.zeros(5)
        for i in ex:
            i_list[i] = 1

        for index, row in df.iterrows():
            if int(index) > idx and len(df.iloc[index]['matching']) > 0:
                if row['careF'] == i_list[0]:
                    if row['fairnessF'] == i_list[1]:
                        if row['loyaltyF'] == i_list[2]:
                            if row['authorityF'] == i_list[3]:
                                if row['purityF'] == i_list[4]:
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
    datasets = ["ALM", "Baltimore", "BLM", "Davidson", "Election", "MeToo", "Sandy"]
    subreddits = ["AmItheAsshole", "antiwork", "confession", "Conservative", "europe", "geopolitics", "neoliberal",
                  "nostalgia", "politics", "relationship_advice", "worldnews"]
    # generate_sup_foundation("annotations/train/both_train.csv")
    generate_sup_foundation("processed/mftc/train/merged_MFTC_train_add.csv")


if __name__ == "__main__":
    main()


