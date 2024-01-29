from sklearn.model_selection import GridSearchCV

# parameters = {
#     "learning_rate": [3e-5],
#     "num_train_epochs": [1, 2, 3],
#     "max_sequence_length": [64, 128],
#     "per_device_train_batch_size": [16, 32],
#     "temp": [0.01, 0.05, 0.1]
# }
#
#
# def write_to_sbatch():
#     models = []
#     for learning_rate in parameters["learning_rate"]:
#         for epoch in parameters["num_train_epochs"]:
#             for max_seq_length in parameters["max_sequence_length"]:
#                 for batch_size in parameters["per_device_train_batch_size"]:
#                     for temp in parameters["temp"]:
#                         models.append(f"model/new-unsup-batch{batch_size}/unsup-large-lr{learning_rate}-ep{epoch}-seq{max_seq_length}-batch{batch_size}-temp{temp}")
#                         srun_text = f"srun python train.py --model_name_or_path princeton-nlp/unsup-simcse-bert-large-uncased " + \
#                         f"--train_file data/shuffled_merged_MFTC_train_add_unsup.txt --output_dir result/shuffled-unsup-batch{batch_size}/unsup-large-lr{learning_rate}-ep{epoch}-seq{max_seq_length}-batch{batch_size}-temp{temp} " + \
#                         f"--num_train_epochs {epoch} --per_device_train_batch_size {batch_size} --learning_rate {learning_rate} --max_seq_length {max_seq_length} " + \
#                         f"--pad_to_max_length --metric_for_best_model stsb_spearman --load_best_model_at_end --pooler_type cls " + \
#                         f"--mlp_only_train --overwrite_output_dir --temp {temp} --do_train"
#                         hugging_text = f"srun python simcse_to_huggingface.py --path result/shuffled-unsup-batch{batch_size}/unsup-large-lr{learning_rate}-ep{epoch}-seq{max_seq_length}-batch{batch_size}-temp{temp}"
#                         with open("unsup_grid_search16.sbatch", "a") as sbatch_file:
#                             sbatch_file.write(srun_text)
#                             sbatch_file.write("\n")
#                             sbatch_file.write(hugging_text)
#                             sbatch_file.write("\n")
#
#     print(models)
#
#
# write_to_sbatch()
# Supervised
parameters = {
    "learning_rate": [5e-5],
    "num_train_epochs": [2, 3, 5],
    "max_sequence_length": [64, 128],
    "per_device_train_batch_size": [16,32],
    "temp": [0.01, 0.05, 0.1]
}

models = []
def write_to_sbatch():

    for learning_rate in parameters["learning_rate"]:
        for epoch in parameters["num_train_epochs"]:
            for max_seq_length in parameters["max_sequence_length"]:
                for batch_size in parameters["per_device_train_batch_size"]:
                    for temp in parameters["temp"]:
                        models.append(
                            f"model/sup-batch{batch_size}/large-lr{learning_rate}-ep{epoch}-seq{max_seq_length}-batch{batch_size}-temp{temp}")
                        srun_text = f"srun python train.py --model_name_or_path princeton-nlp/sup-simcse-bert-large-uncased " + \
                        f"--train_file data/MFTC_supervised.csv --output_dir result/large-lr{learning_rate}-ep{epoch}-seq{max_seq_length}-batch{batch_size}-temp{temp} " + \
                        f"--num_train_epochs {epoch} --per_device_train_batch_size {batch_size} --learning_rate {learning_rate} --max_seq_length {max_seq_length} " + \
                        f"--pad_to_max_length --metric_for_best_model stsb_spearman --load_best_model_at_end --pooler_type cls " + \
                        f"--overwrite_output_dir --temp {temp} --do_train"
                        hugging_text = f"srun python simcse_to_huggingface.py --path result/large-lr{learning_rate}-ep{epoch}-seq{max_seq_length}-batch{batch_size}-temp{temp}"
                        with open("grid_search.sbatch", "a") as sbatch_file:
                            sbatch_file.write(srun_text)
                            sbatch_file.write("\n")
                            sbatch_file.write(hugging_text)
                            sbatch_file.write("\n")

    print(models)

write_to_sbatch()

