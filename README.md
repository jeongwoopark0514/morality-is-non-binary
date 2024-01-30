# Morality is Non-Binary: Building a Pluralist Moral Sentence Embedding Space using Contrastive Learning

Code for the following EACL2024 paper:

Jeongwoo Park, Enrico Liscio, and Pradeep K. Murukannaiah. 2024. Morality is Non-Binary: Building a Pluralist Moral Sentence Embedding Space using Contrastive Learning. In _Findings of the Association for Computational Linguistics: EACL 2024_, ST. Julian's, Malta. Association for Computational Linguistics.

## Preparation

The code is tested in Python 3.9 (pip=21.1.1) Conda environment.

Please download the MFTC dataset to the `data` folder. The trained models used in the paper are available on [$TU.ResearchData](https://doi.org/10.4121/e0d75aad-6cd1-45dd-a5ec-985e399337b4), or can be generated as follows.

## Steps
1. Generate dataset via data folder. (Check readme in data folder)
2. Create SimCSE embeddings. (Check [SimCSE](https://github.com/princeton-nlp/SimCSE) README. I recommend pulling their repository and follow their instruction)
3. Check the output using `finetune/classify.py`. Make sure to input appropriate hyperparameters.

## Set up in HPC Cluster
Due to environment issues in HPC cluster, I recommend the following approach.
1. Create a conda environment.
2. Install pip 21.1.1
3. Follow README of transformer.
4. Install sentencepiece

## Generate embedding space
Details about SimCSE parameters, setup environments can be found in https://github.com/princeton-nlp/SimCSE.
Make sure to run `simcse_to_huggingface.py` after creating the embedding space.

### Command Example for Supervised Learning
``
python train.py --model_name_or_path princeton-nlp/sup-simcse-bert-large-uncased --train_file data/MFTC_supervised.csv --output_dir result/large-lr5e-05-ep2-seq64-batch32-temp0.1 --num_train_epochs 2 --per_device_train_batch_size 32 --learning_rate 5e-05 --max_seq_length 64 --pad_to_max_length --metric_for_best_model stsb_spearman --load_best_model_at_end --pooler_type cls --overwrite_output_dir --temp 0.1 --do_train
``

And then change the SimCSE format to huggingface format.

``
python simcse_to_huggingface.py --path result/large-lr5e-05-ep2-seq64-batch32-temp0.1
``
