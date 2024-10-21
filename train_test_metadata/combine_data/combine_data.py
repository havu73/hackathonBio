import os

import pandas as pd
import numpy as np


vibrio_folder = '/mnt/efs/fs1/data/metadata/vibrio'
ecoli_folder='/mnt/efs/fs1/data/metadata/ecoli'
kleb_folder = '/mnt/efs/fs1/data/metadata/klebsiella'
phageDB_folder = '/mnt/efs/fs1/data/metadata/phageDB'
phageScope_folder = '/mnt/efs/fs1/data/metadata/phageScope'
vikecoli_folder = '/mnt/efs/fs1/data/metadata/vikecoli'
embed_folder = '/mnt/efs/fs1/data/embeddings'
raw_folder = '/mnt/efs/fs1/data/raw'
projects = ['vibrio', 'ecoli', 'klebsiella', 'phageDB', 'phageScope']

def get_negative_pairs(positive_pair_df, num_negative_pairs=1, train_or_test='test'):
    positive_pair_df['label'] = 1
    hosts = positive_pair_df['Host_ID'].unique()
    phages = positive_pair_df['Phage_ID'].unique()
    print(f'Number of phages: {len(phages)} in positive_pair_df')
    print(f'Number of hosts: {len(hosts)} in positive_pair_df')
    neg_test_pair_df = pd.DataFrame(columns=['Phage_ID', 'Host_ID', 'train_or_test'])
    from itertools import product
    all_possible_negative_pairs = set(product(phages, hosts))
    positive_pairs = set(zip(positive_pair_df['Phage_ID'], positive_pair_df['Host_ID']))
    all_possible_negative_pairs = all_possible_negative_pairs - positive_pairs
    all_possible_negative_pairs = list(all_possible_negative_pairs)
    num_negative_pairs = min(num_negative_pairs, len(all_possible_negative_pairs))  # only sample as much as we can
    sample_neg_pairs = np.random.choice(len(all_possible_negative_pairs), num_negative_pairs, replace=False)
    for idx in sample_neg_pairs:
        phage_id, host_id = all_possible_negative_pairs[idx]
        neg_test_pair_df.loc[neg_test_pair_df.shape[0]] = {'Phage_ID': phage_id, 'Host_ID': host_id, 'train_or_test': train_or_test}
    neg_test_pair_df['label'] = 0
    return neg_test_pair_df

def soft_line_combine_embeddings(projects, output_folder):
    '''
    Given the list of projects:
    - soft link all the files from embed_folder/project_embeddings/phage_embeddings/ to output_folder/phage_embeddings/
    - soft link all the files from embed_folder/project_embeddings/host_embeddings/ to output_folder/host_embeddings/
    '''
    os.makedirs(f'{output_folder}/phage_embeddings', exist_ok=True)
    os.makedirs(f'{output_folder}/host_embeddings', exist_ok=True)
    for project in projects:
        os.system(f'ln -s {embed_folder}/{project}_embeddings/phage_embeddings/* {output_folder}/phage_embeddings/')
        os.system(f'ln -s {embed_folder}/{project}_embeddings/host_embeddings/* {output_folder}/host_embeddings/')
    return

def soft_line_combine_fastas(projects, output_folder):
    '''
    Given the list of projects:
    - soft link all the files from embed_folder/project_embeddings/phage_embeddings/ to output_folder/phage_embeddings/
    - soft link all the files from embed_folder/project_embeddings/host_embeddings/ to output_folder/host_embeddings/
    '''
    os.makedirs(f'{output_folder}/phage_fasta', exist_ok=True)
    os.makedirs(f'{output_folder}/host_fasta', exist_ok=True)
    for project in projects:
        os.system(f'ln -s {raw_folder}/{project}/phage_fasta/* {output_folder}/phage_fasta/')
        os.system(f'ln -s {raw_folder}/{project}/host_fasta/* {output_folder}/host_fasta/')
    return

def combine_vikecoli():
    # first, combine positive pairs for vibrio, ecoli, and klebsiella
    output_folder = vikecoli_folder
    # combine train and test positive pairs into one file
    train_df_list = []
    test_df_list = []
    for folder in [vibrio_folder, ecoli_folder, kleb_folder]:
        train_fn = f'{folder}/train_positive_pairs.txt'
        test_fn = f'{folder}/test_positive_pairs.txt'
        train_df = pd.read_csv(train_fn, sep='\t', header=0, index_col=None)
        test_df = pd.read_csv(test_fn, sep='\t', header=0, index_col=None)
        train_df_list.append(train_df)
        test_df_list.append(test_df)
    train_pos_df = pd.concat(train_df_list, axis=0)
    test_pos_df = pd.concat(test_df_list, axis=0)
    train_pos_df.to_csv(f'{output_folder}/train_positive_pairs.txt', sep='\t', index=False)
    test_pos_df.to_csv(f'{output_folder}/test_positive_pairs.txt', sep='\t', index=False)
    # now generate test_prompt and the ground truth files
    num_test_neg_pairs = test_pos_df.shape[0]* 10
    neg_test_df = get_negative_pairs(test_pos_df, num_negative_pairs=num_test_neg_pairs, train_or_test='test')
    neg_test_df['label'] = 0
    test_pos_df['label'] = 1
    test_results_df = pd.concat([test_pos_df, neg_test_df], axis=0)
    test_results_df = test_results_df.reset_index(drop=True)
    test_results_df.to_csv(f'{output_folder}/test_ground_truth.txt', index=False, sep = '\t', header = True)
    test_prompt_df = test_results_df[['Phage_ID', 'Host_ID']]
    test_prompt_df.to_csv(f'{output_folder}/test_prompt.txt', index=False, sep = '\t', header = True)
    output_embed_folder = f'{embed_folder}/vikecoli_embeddings'
    soft_line_combine_embeddings(['vibrio','klebsiella', 'ecoli'], output_embed_folder)
    output_fasta_folder = f'{raw_folder}/vikecoli'
    soft_line_combine_fastas(['vibrio','klebsiella', 'ecoli'], output_fasta_folder)
    return train_df, test_df



train_pos_df, test_pos_df = combine_vikecoli()
