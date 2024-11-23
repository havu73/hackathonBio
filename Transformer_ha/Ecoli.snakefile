import pandas as pd
import numpy as np
import os
phage_embed = '/gladstone/engelhardt/lab/hvu/hackathonBio/Ecoli/phages/embed/'
host_embed = '/gladstone/engelhardt/lab/hvu//hackathonBio/Ecoli/host/embed/chunk_40K_updated/'
meta_folder = '/gladstone/engelhardt/lab/hvu//hackathonBio/Ecoli/models/cross_val/column/'
num_round = 20
output_folder = '/gladstone/engelhardt/lab/hvu//hackathonBio/Ecoli/models/Transformer/chunk_40K/column/'
max_len = 200
input_dim = 4096
model_dim = 1024
num_heads = 8
num_layers = 4
dim_feedforward = 4096
dropout=0.0

model_fn = '/gladstone/engelhardt/lab/hvu/hackathonBio/hackathon_results/tung/tung_results/ecoli/best_checkpoint.pth'
norm_fn = '/gladstone/engelhardt/lab/hvu/hackathonBio/hackathon_results/tung/tung_results/ecoli/norm_constants.pth'

rule all:
    input:
        expand(os.path.join(output_folder, 'round_{i}', 'predict.txt'), i=range(num_round))

rule train:
    input:
        os.path.join(meta_folder, 'round_{i}', 'train_positive_pairs.txt'),
        os.path.join(meta_folder, 'round_{i}', 'train_negative_pairs.txt')
    output:
        os.path.join(output_folder, 'round_{i}', 'model.txt')
    shell:
        '''
        python train.py \
            --phage_embed {phage_embed} \
            --host_embed {host_embed} \
            --train_positive_fn  {input[0]} \
            --train_negative_fn {input[1]} \
            --save_fn  {output}
        '''

rule predict:
    input:
        model_fn,
        norm_fn,
        os.path.join(meta_folder, 'round_{i}', 'test_prompt.txt'),

    output:
        os.path.join(output_folder, 'round_{i}', 'predict.txt')
    shell:
        '''
        python test.py \
            --phage_embed {phage_embed} \
            --host_embed {host_embed} \
            --model_fn {input[0]} \
            --test_prompt_fn  {input[2]} \
            --save_fn  {output} \
            --norm_constants_fn {input[1]}
        '''
