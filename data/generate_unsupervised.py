import numpy as np
import pandas as pd

datasets = ["ALM", "Baltimore", "BLM", "Davidson", "Election", "MeToo", "Sandy"]
subreddits = ["AmItheAsshole", "antiwork", "confession", "Conservative", "europe", "geopolitics", "neoliberal", "nostalgia", "politics", "relationship_advice", "worldnews"]

def generate_unsup(file_path: str, trainOrTest: bool):
    df = pd.read_csv(file_path)
    output_df = df['processed']

    if trainOrTest:
        # np.savetxt(file_path.split('.')[0] + "_unsup.txt", df.values)
        output_df.to_csv(file_path.split('.')[0] + "_unsup.txt", index=False, header=False)


if __name__ == '__main__':
    unsup_files = ["processed/mftc/5fold/train_set1.csv", "processed/mftc/5fold/train_set2.csv", "processed/mftc/5fold/train_set3.csv", "processed/mftc/5fold/train_set4.csv", "processed/mftc/5fold/train_set5.csv"]
    for i in range(len(unsup_files)):
        generate_unsup(unsup_files[i], True)