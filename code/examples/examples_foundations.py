import pandas as pd

def check_label(item1, item2):
    if item1["care"] == item2["care"] and item1["fairness"] == item2["fairness"] and item1["loyalty"] == item2["loyalty"] and \
        item1["authority"] == item2["authority"] and item1["purity"] == item2["purity"] and item1["harm"] == item2["harm"] and \
            item1["cheating"] == item2["cheating"] and item1["betrayal"] == item2["betrayal"] and item1["subversion"] == item2["subversion"] and \
            item1["degradation"] == item2["degradation"]:
        return True

    else:
        # print(item1, item2)
        return False


top1_count, top2_count, top3_count = 0, 0, 0

df = pd.read_csv("sup_examples_similar_sentences.csv",
                 sep=',',
                 names=["query", "top1", "top1score", "top2", "top2score", "top3", "top3score"])

train_df = pd.read_csv("../examples/data/merged_MFTC_train_add.csv")
test_df = pd.read_csv("../examples/data/merged_MFTC_test_add.csv")

for index, row in df.iterrows():
    query = row["query"]
    top1 = row["top1"]
    top2 = row["top2"]
    top3 = row["top3"]

    query_df = test_df[test_df["processed"] == query]
    top1_df = train_df[train_df["processed"] == top1]
    top2_df = train_df[train_df["processed"] == top2]
    top3_df = train_df[train_df["processed"] == top3]

    top1_label = False
    top2_label = False
    top3_label = False


    for index0, row0 in query_df.iterrows():
        for index1, row1 in top1_df.iterrows():
            if check_label(row0, row1):
                top1_label = True
                break

        for index2, row2 in top2_df.iterrows():
            if check_label(row0, row2):
                top2_label = True
                break

        for index3, row3 in top3_df.iterrows():
            if check_label(row0, row3):
                top3_label = True
                break

        if top1_label:
            top1_count += 1
        if top2_label:
            top2_count += 1
        if top3_label:
            top3_count += 1

print("Total: ", len(df))
print("Same top1 label: ", top1_count)
print("Same top2 label: ", top2_count)
print("Same top3 label: ", top3_count)
