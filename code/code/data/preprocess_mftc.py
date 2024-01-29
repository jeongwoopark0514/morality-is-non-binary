import sys
import pandas as pd
from tqdm import tqdm
import json
from cleaners import cleaner1, cleaner2, cleaner3, cleaner4, cleaner5

DUMMY_TEXT = "no tweet text available"

output = {"tweet_id": [], "text": [], "fairness": [], "non-moral": [], "purity": [], "degradation": [], "loyalty": [],
          "care": [], "cheating": [], "betrayal": [], "subversion": [], "authority": [], "harm": []}

moralValues = ["fairness", "non-moral", "purity", "degradation", "loyalty", "care", "cheating", "betrayal",
               "subversion", "authority", "harm"]


def getValues(selectedAnnotations):
    values = []
    for value in moralValues:
        values.append((value, int(value in selectedAnnotations)))
    return values


def annotationStrategy0(annotations):
    selectedAnnotations = ",".join([annotator['annotation'] for annotator in annotations])
    return getValues(selectedAnnotations)


def annotationStrategy1(annotations):
    majorityAnnotations = []
    addedValues = (",".join([annotator['annotation'] for annotator in annotations])).split(",")
    for value in moralValues:
        if addedValues.count(value) >= len(annotations) / 2:
            majorityAnnotations.append(value)
    if 'non-moral' in majorityAnnotations and len(majorityAnnotations) > 1:
        majorityAnnotations.remove('non-moral')
    selectedAnnotations = ",".join(majorityAnnotations)
    if selectedAnnotations == '':
        selectedAnnotations = 'non-moral'
    return getValues(selectedAnnotations)


def preprocessStrategy0(rawText):
    return rawText


def preprocessStrategy1(rawText):
    return cleaner1(rawText)


def preprocessStrategy2(rawText):
    return cleaner2(rawText)


def preprocessStrategy3(rawText):
    return cleaner3(rawText)


def preprocessStrategy4(rawText):
    return cleaner4(rawText)


# This is the strategy used in our experiments.
def preprocessStrategy5(rawText):
    return cleaner5(rawText)


preprocess = {"0": preprocessStrategy0, "1": preprocessStrategy1, "2": preprocessStrategy2, "3": preprocessStrategy3,
              "4": preprocessStrategy4, "5": preprocessStrategy5}


def remove_dup_final(file_path: str, corpus_name: str):
    df = pd.read_csv(file_path)
    df['count'] = 1

    if corpus_name == 'all':
        crit = 'processed'
    else:
        crit = 'text'
    the_group = df.groupby(crit, as_index=False)
    df = the_group.agg({'fairness':'sum', 'non-moral':'sum', 'purity': 'sum', 'degradation': 'sum', 'loyalty': 'sum', 'care': 'sum', 'cheating': 'sum', 'betrayal': 'sum', 'subversion': 'sum', 'authority': 'sum', 'harm': 'sum', 'count': 'sum'})

    for index, row in df.iterrows():
        count = row['count']
        if count > 1:
            major = []
            for i in range(len(moralValues)):
                if row[moralValues[i]] >= count / 2:
                    df.at[index, moralValues[i]] = 1
                    # row[moralValues[i]] = 1
                    major.append(moralValues[i])
                else:
                    df.at[index, moralValues[i]] = 0
            if ('non-moral' in major and len(major) > 1 ) or len(major) < 1:
                df.at[index, 'non-moral'] = 1
                for i in range(len(moralValues)):
                    df.at[index, moralValues[i]] = 0

            for i in range(len(moralValues)):
                df.at[index, 'count'] += df.iloc[index][moralValues[i]]
        else:
            continue
    df = df.rename(columns={"text":"processed"})
    df.to_csv(file_path.split('.')[0] + "_add.csv", index=False)


if __name__ == '__main__':
    # path = sys.argv[1]
    # strategy = sys.argv[2]

    # Clean up preprocessed data
    datasets = ["ALM", "Baltimore", "BLM", "Davidson", "Election", "MeToo", "Sandy"]
    for i in range(len(datasets)):
        remove_dup_final(f"processed/mftc/{datasets[i]}.csv", datasets[i])
