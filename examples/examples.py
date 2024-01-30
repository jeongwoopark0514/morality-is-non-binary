import csv
import pandas as pd
from simcse import SimCSE


## Supervised
# Build index
file_path = "../examples/data/merged_MFTC_train_add_unsup.txt"
supervised_simcse_model = SimCSE("../SimCSE/result/sup-batch32/large-lr5e-05-ep2-seq64-batch32-temp0.1")
supervised_simcse_model.build_index(file_path, batch_size=32)

# Read in the test file
test_file_path = "../examples/data/merged_MFTC_test_add.csv"
test_file = pd.read_csv(test_file_path)
test_file_texts = test_file["processed"].tolist()

# Search and write it into csv file
with open("sup_examples_similar_sentences.csv", "w", encoding="UTF-8") as file:
    writer = csv.writer(file)

    for sentence in test_file_texts:
        results = [sentence]
        output = supervised_simcse_model.search(sentence, top_k=3)
        for ou in output:
            results.append(ou[0])
            results.append(ou[1])
        writer.writerow(results)

## Unsupervised
# Build index
file_path = "../examples/data/merged_MFTC_train_add_unsup.txt"
unsup_simcse_model = SimCSE("../SimCSE/result/unsup-batch32/unsup-large-lr3e-05-ep1-seq64-batch32-temp0.01")
unsup_simcse_model.build_index(file_path, batch_size=32)

# Read in the test file
test_file_path = "../examples/data/merged_MFTC_test_add.csv"
test_file = pd.read_csv(test_file_path)
test_file_texts = test_file["processed"].tolist()

# Search and write it into csv file
with open("unsup_examples_similar_sentences.csv", "w", encoding="UTF-8") as file:
    writer = csv.writer(file)

    for sentence in test_file_texts:
        results = [sentence]
        output = unsup_simcse_model.search(sentence, top_k=3)
        for ou in output:
            results.append(ou[0])
            results.append(ou[1])
        writer.writerow(results)


