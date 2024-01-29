import pandas as pd
import numpy as np
from simcse import SimCSE


# simcse_model = SimCSE("model/sup-batch32/large-lr5e-05-ep2-seq64-batch32-temp0.1")
simcse_model = SimCSE("model/11label-large-batch16-len64")

# In[4]:


# Read in single labels file for train set and test set.
mftc_train_set = pd.read_csv("data/single_labels.csv", index_col=0)
mftc_test_set = pd.read_csv("data/single_labels_test.csv", index_col=0)


# First process train set
mftc_train_care_df = mftc_train_set[mftc_train_set["care"] == 1]
mftc_train_harm_df = mftc_train_set[mftc_train_set["harm"] == 1]
mftc_train_fairness_df = mftc_train_set[mftc_train_set["fairness"] == 1]
mftc_train_cheating_df = mftc_train_set[mftc_train_set["cheating"] == 1]
mftc_train_loyalty_df = mftc_train_set[mftc_train_set["loyalty"] == 1]
mftc_train_betrayal_df = mftc_train_set[mftc_train_set["betrayal"] == 1]
mftc_train_authority_df = mftc_train_set[mftc_train_set["authority"] == 1]
mftc_train_subversion_df = mftc_train_set[mftc_train_set["subversion"] == 1]
mftc_train_purity_df = mftc_train_set[mftc_train_set["purity"] == 1]
mftc_train_degradation_df = mftc_train_set[mftc_train_set["degradation"] == 1]
mftc_train_non_moral_df = mftc_train_set[mftc_train_set["non-moral"] == 1]


# In[10]:


# First process test set
mftc_test_care_df = mftc_test_set[mftc_test_set["care"] == 1]
mftc_test_harm_df = mftc_test_set[mftc_test_set["harm"] == 1]
mftc_test_fairness_df = mftc_test_set[mftc_test_set["fairness"] == 1]
mftc_test_cheating_df = mftc_test_set[mftc_test_set["cheating"] == 1]
mftc_test_loyalty_df = mftc_test_set[mftc_test_set["loyalty"] == 1]
mftc_test_betrayal_df = mftc_test_set[mftc_test_set["betrayal"] == 1]
mftc_test_authority_df = mftc_test_set[mftc_test_set["authority"] == 1]
mftc_test_subversion_df = mftc_test_set[mftc_test_set["subversion"] == 1]
mftc_test_purity_df = mftc_test_set[mftc_test_set["purity"] == 1]
mftc_test_degradation_df = mftc_test_set[mftc_test_set["degradation"] == 1]
mftc_test_non_moral_df = mftc_test_set[mftc_test_set["non-moral"] == 1]


def similarity_and_text(model, text_df1, text_df2, save_name):
    df_output = {
        "text1": [],
        "text2": [],
        "similarity": []
    }
    sim_results = model.similarity(text_df1, text_df2)
    for i in range(len(text_df1)):
        for j in range(len(text_df2)):
            df_output["text1"].append(text_df1[i])
            df_output["text2"].append(text_df2[j])
            df_output["similarity"].append(sim_results[i][j])

    df = pd.DataFrame(df_output)
    print(df)
    df.to_csv(save_name, index=False)


def find_common_lemma(text_df1_list, text_df2_list, save_file_name):
    # import these modules
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    from nltk import FreqDist, download
    from nltk.tokenize import word_tokenize
    download('omw-1.4')

    # First Tokenize
    flat_list = [sublist for sublist in text_df1_list]
    tokenized_list = []
    for sentence in flat_list:
        tokenized_list.append(word_tokenize(sentence))

    # First remove stop words
    stop_words = stopwords.words('english')
    # Use a list comprehension to remove stop words
    filtered_list = [list(set(sublist).difference(stop_words)) for sublist in tokenized_list]

    # print(filtered_list)

    # First get lemma
    lemmatizer = WordNetLemmatizer()
    lemmatized = [[lemmatizer.lemmatize(word) for word in s]
                  for s in filtered_list]
    # Second Tokenize
    flat_list2 = [sublist for sublist in text_df2_list]
    tokenized_list2 = []
    for sentence in flat_list2:
        tokenized_list2.append(word_tokenize(sentence))

    # Second remove stop words
    # Use a list comprehension to remove stop words
    filtered_list2 = [list(set(sublist).difference(stop_words)) for sublist in tokenized_list2]
    # print(filtered_list)

    # Get lemma
    lemmatized_2 = [[lemmatizer.lemmatize(word) for word in s]
                  for s in filtered_list2]
    full_lemmatized = lemmatized + lemmatized_2

    allWordDist1 = FreqDist(w for sublist in lemmatized for w in sublist)
    mostCommon1 = allWordDist1.most_common(100)
    print("1: ", mostCommon1)

    allWordDist2 = FreqDist(w for sublist in lemmatized_2 for w in sublist)
    mostCommon2 = allWordDist2.most_common(100)
    print("2: ", mostCommon2)

    # allWordDist = FreqDist(w for sublist in full_lemmatized for w in sublist)
    # mostCommon_full = allWordDist.most_common(20)
    # save_most_common(mostCommon1, mostCommon2, mostCommon_full, save_file_name)
    # print("Full: ", mostCommon_full)
    mostCommon1_dict = dict(mostCommon1)
    mostCommon2_dict = dict(mostCommon2)
    mostCommon1Keys = mostCommon1_dict.keys()
    mostCommon2Keys = mostCommon2_dict.keys()
    # mostCommon_full = set(mostCommon1Keys).intersection(set(mostCommon2Keys))
    #
    # Make sure to keep order using the first moral value freq counts
    set_2 = frozenset(mostCommon2Keys)
    mostCommon_full = [x for x in mostCommon1Keys if x in set_2]

    save_most_common(mostCommon1, mostCommon2, mostCommon_full, save_file_name)
    print("Full: ", mostCommon_full)


    # save_most_common(mostCommon1, mostCommon2, mostCommon_full, save_file_name)
    # print("Full: ", mostCommon_full)



def save_most_common(comm_list1, comm_list2, comm_list3, file_name):
    dict_to_df = {
        "words1": [],
        "counts1": [],
        "words2": [],
        "counts2": [],
        "wordsFull": [],
    }
    for item in comm_list1:
        dict_to_df["words1"].append(item[0])
        dict_to_df["counts1"].append(int(item[1]))
    for item in comm_list2:
        dict_to_df["words2"].append(item[0])
        dict_to_df["counts2"].append(int(item[1]))
    for item in comm_list3:
        dict_to_df["wordsFull"].append(item)
    df = pd.DataFrame.from_dict(dict_to_df, orient='index')
    df = df.transpose()
    df.to_csv(f"{file_name}.csv", index=False)




# similarity_and_text(simcse_model, mftc_train_fairness_df["processed"].tolist(), mftc_train_authority_df["processed"].tolist(), "fairness_authority_train.csv")
# similarity_and_text(simcse_model, mftc_test_fairness_df["processed"].tolist(), mftc_test_authority_df["processed"].tolist(), "fairness_authority_test.csv")
# find_common_lemma(mftc_train_care_df["processed"].tolist(), mftc_train_purity_df["processed"].tolist(), save_file_name="care_purity_all")
# find_common_lemma(mftc_train_fairness_df["processed"].tolist(), mftc_train_authority_df["processed"].tolist(), save_file_name="fairness_authority_all")
# find_common_lemma(mftc_train_fairness_df["processed"].tolist(), mftc_train_cheating_df["processed"].tolist(), save_file_name="fairness_cheating_all")
# find_common_lemma(mftc_train_authority_df["processed"].tolist(), mftc_train_subversion_df["processed"].tolist(), save_file_name="authority_subversion_all")
find_common_lemma(mftc_train_betrayal_df["processed"].tolist(), mftc_train_subversion_df["processed"].tolist(), save_file_name="betrayal_subversion_all")