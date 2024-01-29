import numpy as np
import pandas as pd

# datasets_train = ["processed/mftc/train/ALM_train_sup.csv", "processed/mftc/train/Baltimore_train_sup.csv", "processed/mftc/train/BLM_train_sup.csv",
#             "processed/mftc/train/Davidson_train_sup.csv", "processed/mftc/train/Election_train_sup.csv", "processed/mftc/train/MeToo_train_sup.csv",
#             "processed/mftc/train/Sandy_train_sup.csv"]
#
# datasets_advanced_train = ["processed/mftc/train/ALM_train_advanced_sup.csv", "processed/mftc/train/Baltimore_train_advanced_sup.csv", "processed/mftc/train/BLM_train_advanced_sup.csv",
#             "processed/mftc/train/Davidson_train_advanced_sup.csv", "processed/mftc/train/Election_train_advanced_sup.csv", "processed/mftc/train/MeToo_train_advanced_sup.csv",
#             "processed/mftc/train/Sandy_train_advanced_sup.csv"]

# datasets_test = ["processed/mftc/test/ALM_test.csv", "processed/mftc/test/Baltimore_test.csv", "processed/mftc/test/BLM_test.csv",
#             "processed/mftc/test/Davidson_test.csv", "processed/mftc/test/Election_test.csv", "processed/mftc/test/MeToo_test.csv",
#             "processed/mftc/test/Sandy_test.csv"]

datasets = ["ALM", "Baltimore", "BLM", "Davidson", "Election", "MeToo", "Sandy"]
subreddits = ["AmItheAsshole", "antiwork", "confession", "Conservative", "europe", "geopolitics", "neoliberal", "nostalgia", "politics", "relationship_advice", "worldnews"]
both = ["MFTC", "mfrc"]
moralValues = ["fairness", "non-moral", "purity", "degradation", "loyalty", "care", "cheating", "betrayal",
               "subversion", "authority", "harm"]

def generate_foundation_columns(df: pd.DataFrame):
    dff = df.copy()
    dff['careF'] = dff['care'] + dff['harm']
    dff['fairnessF'] = dff['fairness'] + dff['cheating']
    dff['loyaltyF'] = dff['loyalty'] + dff['betrayal']
    dff['authorityF'] = dff['authority'] + dff['subversion']
    dff['purityF'] = dff['purity'] + dff['degradation']

    dff['careF'] = np.where(dff['careF'] > 1, 1, dff['careF'])
    dff['fairnessF'] = np.where(dff['fairnessF'] > 1, 1, dff['fairnessF'])
    dff['loyaltyF'] = np.where(dff['loyaltyF'] > 1, 1, dff['loyaltyF'])
    dff['authorityF'] = np.where(dff['authorityF'] > 1, 1, dff['authorityF'])
    dff['purityF'] = np.where(dff['purityF'] > 1, 1, dff['purityF'])
    dff['count'] = dff['careF'] + dff['fairnessF'] + dff['loyaltyF'] + dff['authorityF'] + dff['purityF']

    return dff

def merge_csv_files(mftc_mfrc: bool, train_or_test: bool, sup_or_unsup: bool):
    to_merge = []
    domains = datasets if mftc_mfrc else subreddits
    for i in range(len(domains)):
        dataset = "mftc" if mftc_mfrc else "mfrc"
        slot1 = "train" if train_or_test else "test"
        slot2 = "supervised" if sup_or_unsup else "unsupervised"
        suffix = "advanced_sup.csv" if sup_or_unsup else "unsup.txt"
        to_merge.append(f"processed/{dataset}/{slot1}/{slot2}/{domains[i]}_{slot1}_{suffix}")

    if sup_or_unsup:
        df = pd.concat(
            map(pd.read_csv, to_merge), ignore_index=True)

        return df.to_csv("mfrc_test_supervised.csv", index=False)
    else:
        df_list = []
        for i in range(len(to_merge)):
            df_list.append(pd.read_csv(to_merge[i], header=None))

        big_df = pd.concat(df_list, ignore_index=True)
        big_df.to_csv("mfrc_train_unsupervised.txt", header=False, index=False)

def mergeCorpuses():
    combined_csv = pd.concat([pd.read_csv('processed/mftc/' + f'{corpus}_add.csv') for corpus in datasets])
    # export all to csv
    combined_csv.to_csv(f'processed/mftc/merged_MFTC.csv', index=False, encoding='utf-8-sig')


def merge_mftc_mfrc(mftc_filepath, mfrc_filepath, file_name):
    mftc_file = pd.read_csv(mftc_filepath)
    mfrc_file = pd.read_csv(mfrc_filepath)

    output = {"processed": [], "care": [], "harm": [], "fairness": [],
              "cheating": [], "loyalty": [], "betrayal": [], "authority": [], "subversion": [], "purity": [],
              "degradation": [], "non-moral": []}

    output['processed'] = mftc_file['processed'].values.tolist()
    output['processed'].extend(mfrc_file['processed'].values.tolist())
    #
    for i in range(len(moralValues)):
        output[moralValues[i]] = mftc_file[moralValues[i]].values.tolist()
        output[moralValues[i]].extend(mfrc_file[moralValues[i]].values.tolist())

    output_df = pd.DataFrame(output)
    return output_df.to_csv(f"processed/both/{file_name}.csv", index=False)


def main():
    # combined_csv = pd.concat([pd.read_csv("./processed/mftc/train/all_corpus_train_temp1_exclude_outside_foundation_shuffle.csv"),
    #                           pd.read_csv("./processed/mftc/train/all_corpus_train_temp2_exclude_within_foundation_shuffle.csv")])
    # # export all to csv
    # combined_csv.to_csv(f'./processed/mftc/train/merged_train_sup_half.csv', index=False, encoding='utf-8-sig')

    unsup_txt = pd.read_csv("./processed/mftc/train/unsupervised/merged_MFTC_train_add_unsup.txt")
    unsup_txt = unsup_txt.sample(frac=1, random_state=42).reset_index(drop=True)
    unsup_txt.to_csv("./processed/mftc/train/unsupervised/shuffled_merged_MFTC_train_add_unsup.txt")
    # merge_mftc_mfrc("./processed/both/MFTC_all_test.csv", "./processed/both/mfrc_all_test.csv", "both_test")
    # merge_csv_files(False, False, False)

if __name__ == "__main__":
    main()
