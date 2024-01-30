import argparse
import pandas as pd
import torch
from simcse import SimCSE

# This code is to generate the sentence embedding in bulk for experiments.
# https://github.com/sent-subsent-embs/clustering-network-analysis/blob/main/src/embedding/src/sentbert/build_sentbert_space.py
MORAL_FOUNDATIONS = {
    'care': 'harm',
    'fairness': 'cheating',
    'loyalty': 'betrayal',
    'authority': 'subversion',
    'purity': 'degradation',
    'non-moral': 'non-moral'
}


def transform_sentences(_sent_map, model_name, batch_size, max_length):
    """
    In the sentence space, we rely entirely on the machinery of the language model, thus there is no way to
    tune the dimensionality of the embeddings. If NOT using sentence transformer and using a raw language model,
    the options for compressing the output to one vector are as follows:
        1. Use the vector associated with the [CLS] token.
        2. Max-pool across each dimension of all output vectors in the tensor.
        3. Take the average across all dimensions for all output vectors in the tensor.
    """
    model = SimCSE(model_name)
    embeddings = model.encode(_sent_map, keepdim=True, batch_size=batch_size, max_length=max_length).numpy()
    print("_sent_map size: ", len(_sent_map))
    print("embeddings: ", embeddings.shape)
    #
    # sentences = list(_sent_map.keys())
    # _sent_embs = model.encode(sentences, show_progress_bar=True)
    # _sent_tensors = [torch.from_numpy(j) for j in _sent_embs]
    return _sent_map, embeddings



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="build_sentbert_space",
                                     description="Builds sentence embeddings from the sentbert model")
    parser.add_argument('--model_name', default=None, type=str,
                        help='model name')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size')
    parser.add_argument('--max_length', default=64, type=int,
                        help='max length')
    parser.add_argument('--data_file', default='context', type=str,
                        help='input data type')
    parser.add_argument('--save_file', default='save', type=str,
                        help='save file name')

    args = parser.parse_args()
    # with open(args.data_file, 'r', encoding="UTF-8") as f:
    #     sentences = f.read().splitlines()

    # For MFTC,
    csv_file = pd.read_csv(args.data_file)
    sentences = csv_file["processed"].tolist()

    # csv_file = pd.read_csv(args.data_file, sep='\t')
    # sentences = csv_file.iloc[:, 1].tolist()

    texts, res = transform_sentences(sentences, args.model_name, args.batch_size, args.max_length)
    torch.save(res, 'embedding_space_{d}_sentbert_space.pt'.format(d=args.save_file))
    torch.save(texts, '{d}_texts.pt'.format(d=args.save_file))
    # torch.save(res, '{d}_res.pt'.format(d=args.model_name.split("\\")[1]))