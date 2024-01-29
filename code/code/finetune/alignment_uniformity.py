import pandas as pd
import logging
from simcse import SimCSE
import torch
import torch.nn as nn

# Reference: https://github.com/princeton-nlp/SimCSE/issues/120

mftc_moral_values = ["fairness", "non-moral", "purity", "degradation", "loyalty", "care", "cheating", "betrayal",
               "subversion", "authority", "harm"]

moral_foundations = ['care', 'fairness', 'loyalty', 'authority', 'purity', 'non-moral']


DEFAULT_CONFIG = {
    "simcse_model": ["model/sup-batch32/large-lr5e-05-ep2-seq64-batch32-temp0.1", "model/new-unsup-batch32/unsup-large-lr3e-05-ep1-seq64-batch32-temp0.05"],
    "data_file": "data/align_uniform_test.csv",
    "input_size": 1024,
    "max_length": 64,
    "label_names": mftc_moral_values,
    "use_foundations": False,
    "epochs": 10,
    "loss_fct": nn.BCEWithLogitsLoss(),
    "threshold": 0,
    "batch_size": 16,
    "learning_rate": 0.01,
    "dropout": 0.1,
    "save_path": "classifiers/classifier_test.pt",
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
}

def alignment_uniformity(all_enc1, all_enc2):
    def _norm(x, eps=1e-8):
        xnorm = torch.linalg.norm(x, dim=-1)
        xnorm = torch.max(xnorm, torch.ones_like(xnorm) * eps)
        return x / xnorm.unsqueeze(dim=-1)


    # from Wang and Isola (with a bit of modification)
    # only consider pairs with gs > 4 (from footnote 3)
    def _lalign(x, y, alpha=2):
        return (_norm(x) - _norm(y)).norm(p=2, dim=1).pow(alpha).mean()


    def _lunif(x, t=2):
        sq_pdist = torch.pdist(_norm(x), p=2).pow(2)
        return sq_pdist.mul(-t).exp().mean().log()


    align = _lalign(
        all_enc1,
        all_enc2).item()

    # consider all sentences (from footnote 3)
    unif = _lunif(torch.cat((all_enc1, all_enc2))).item()
    logging.info(f'align {align}\t\t uniform {unif}')


# Generate supervised dataset from the test portion
df = pd.read_csv(DEFAULT_CONFIG["data_file"])

# TODO: VAlidate the dataset

# Calculate the averaged alignment score and uniformity score
batch1 = df["sent0"].to_list()
batch2 = df["sent1"].to_list()
print(type(batch1), type(batch2))


for i in range(len(DEFAULT_CONFIG["simcse_model"])):
    simcse_model = SimCSE(DEFAULT_CONFIG["simcse_model"][i], device=DEFAULT_CONFIG["device"])
    all_enc1 = simcse_model.encode(batch1, batch_size=DEFAULT_CONFIG["batch_size"], max_length=DEFAULT_CONFIG['max_length'])
    print(all_enc1.shape)
    all_enc2 = simcse_model.encode(batch2, batch_size=DEFAULT_CONFIG["batch_size"], max_length=DEFAULT_CONFIG['max_length'])

    alignment_uniformity(all_enc1, all_enc2)