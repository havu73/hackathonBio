import os
import argparse
import importlib
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from dataset import PairDataset
import model
importlib.reload(model)
from modelW import TransformerClassifier
import helper

# Parameters for manual modification in the notebook
negative_fold = 10
phage_embed = '/gladstone/engelhardt/lab/hvu/hackathonBio/Ecoli/phages/embed'
host_embed = '/gladstone/engelhardt/lab/hvu/hackathonBio/Ecoli/host/embed/chunk_40K'
test_prompt_fn = '/gladstone/engelhardt/lab/hvu/hackathonBio/metadata/ecoli/train_positive_pairs.txt'
save_fn = None
model_fn = '/gladstone/engelhardt/lab/hvu/hackathonBio/Ecoli/models/Transformer/chunk_40K/exp2/exp2/checkpoints/best_checkpoint.pth'
norm_constants_fn = '/gladstone/engelhardt/lab/hvu/hackathonBio/hackathon_results/tung/tung_results/ecoli/norm_constants.pth'
max_len = 200
input_dim = 4096
model_dim = 1024
num_heads = 8
num_layers = 4
dim_feedforward = 4096
dropout = 0.0
device = 'cuda'

pairs_df = pd.read_csv(test_prompt_fn, sep='\t')
pairs = [(row['Host_ID'], row['Phage_ID']) for _, row in pairs_df.iterrows()]
dummy_labels = [-1] * len(pairs)

test_dataset = PairDataset(
    host_root=host_embed,
    phage_root=phage_embed,
    pairs=pairs,
    labels=dummy_labels,
    max_len=max_len,
    remove_non_exist=False,
)
norm_constants = torch.load(norm_constants_fn)
test_dataset.set_norm_constants(*norm_constants)

# load model from checkpoints/best_checkpoint.pth
model = TransformerClassifier(
    input_dim=input_dim,
    model_dim=model_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    max_len=max_len,
).cuda()

checkpoint = torch.load(model_fn)

msg = model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model.eval()
print(msg)


# Forward pass
with torch.no_grad():
    embeddings, label, host_lens, phage_lens = test_dataset[1000]
    print(embeddings.shape, label,  host_lens, phage_lens)
    embeddings = embeddings.unsqueeze(0).cuda()
    host_lens = torch.tensor([host_lens], dtype=torch.long).cuda()
    phage_lens = torch.tensor([phage_lens], dtype=torch.long).cuda()
    outputs = model(embeddings, host_lens, phage_lens, need_weights=True)


import pdb; pdb.set_trace()
t = model.attention_weights
print(t[0])
print(t[1])
print(t[2])
print(t[3])