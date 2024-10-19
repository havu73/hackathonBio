import os
import gradio as gr
import pandas as pd
import numpy as np
import torch
from glob import glob
from model import TransformerClassifier

dataset = 'vikecoli'
max_len = 200 if dataset in ['vikecoli', 'vibrio', 'klebsiella', 'ecoli'] else 300

embedding_dir = f'/mnt/efs/fs1/data/embeddings/{dataset}_embeddings'
host_paths = glob(f'{embedding_dir}/host_embeddings/*.pt')
phage_paths = glob(f'{embedding_dir}/phage_embeddings/*.pt')

test_prompt_path =f'/mnt/efs/fs1/data/metadata/{dataset}/test_prompt.txt'
test_df = pd.read_csv(test_prompt_path, sep='\t')
host_names = test_df['Host_ID'].unique().tolist()
host_names = [name for name in host_names if f'{embedding_dir}/host_embeddings/{name}.pt' in host_paths]
phage_names = test_df['Phage_ID'].unique().tolist()
phage_names = [name for name in phage_names if f'{embedding_dir}/phage_embeddings/{name}.pt' in phage_paths]

model = TransformerClassifier(
    input_dim=4096,
    model_dim=1024,
    num_heads=8,
    num_layers=4,
    dim_feedforward=4096,
    max_len=max_len,
)
root_dir = f'/mnt/efs/fs1/tung_results/{dataset}/{dataset}_default/'
ckpt_path = os.path.join(root_dir, 'checkpoints', 'best_checkpoint.pth')
checkpoint = torch.load(ckpt_path)
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model.eval()

# load norm constants
norm_constants = torch.load(os.path.join(root_dir, 'norm_constants.pth'))
host_mean, host_std, phage_mean, phage_std = norm_constants

def predict(host_name, phage_name):
    if host_name is None:
        host_name = 'GCA_002876945.2'
    if phage_name is None:
        phage_name = 'GCA_003931275.1'

    # load host and phage embeddings
    host_embedding = torch.load(f'{embedding_dir}/host_embeddings/{host_name}.pt')
    host_embedding = (host_embedding - host_mean) / host_std
    
    phage_embedding = torch.load(f'{embedding_dir}/phage_embeddings/{phage_name}.pt')
    phage_embedding = (phage_embedding - phage_mean) / phage_std
    
    host_phage = torch.cat([host_embedding, phage_embedding], dim=0)
    pads = torch.zeros(max_len - host_phage.size(0), host_phage.size(1))
    host_phage = torch.cat([host_phage, pads], dim=0)
    
    pred_prob = model(
        host_phage.unsqueeze(0),
        torch.tensor([host_embedding.size(0)]),
        torch.tensor([phage_embedding.size(0)])
    )
    pred_prob = pred_prob.squeeze().item()
    pred_prob = round(pred_prob, 4)
    
    return pred_prob

title = "Phage-Bacteria Interaction Prediction"
description = """
<div style="text-align: center;">
    <p>This model predicts the probability of interaction between a bacteria and a phage.</p>
    <img src="https://i.postimg.cc/Y9kCy7jp/logo.png" width="200px" style="display: block; margin-left: auto; margin-right: auto;">
</div>
"""


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(host_names, label='Bacteria Name'),
        gr.Dropdown(phage_names, label='Phage Name'),
    ],
    outputs=[
        gr.Textbox(label='Matching Probability')
    ],
    title=title,
    description=description,
)

demo.launch(share=True)