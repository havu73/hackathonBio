# %%
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, average_precision_score, roc_auc_score
import matplotlib.pyplot as plt
from time import time
import os
import numpy as np
import pandas as pd
import random
from sklearn.cluster import KMeans
from torch.optim.optimizer import required
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,log_loss
import lightgbm as lgb
import argparse
import glob
import itertools
def parse_arguments():
    """Function to parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simple calculator using argparse")
    
    # Add arguments
    parser.add_argument('--negative_fold', type=int, default=10, help="negative/positive pair ratio")
    parser.add_argument('--phage_embed', type=str, default='phage_embeddings', help="path to phage embeddings folder")
    parser.add_argument('--host_embed', type=str, default='host_embeddings', help="path to host embeddings folder")
    parser.add_argument('--train_positive_fn', type=str, default='train_positive_pairs.txt', help="path to positive pairs")
    parser.add_argument('--train_negative_fn', type=str, default=None, help="path to negative pairs")
    parser.add_argument('--save_fn', type=str, required=True, help="path to save the model")
    # Parse the arguments and return them
    return parser.parse_args()

def process_dict_embed(embed_folder):
    '''
    Given a folder of embeddings, <phage/bacteria_ID>.pt files, return a dictionary of embeddings
    key: <phage/bacteria_ID>,
    value: embedding tensor
    '''
    embed_dict = {}
    for embed_fn in glob.glob(f'{embed_folder}/*.pt'):
        embed_id = embed_fn.split('/')[-1].split('.pt')[0]
        embed_dict[embed_id] = torch.load(f'{embed_fn}').sum(dim=0,keepdim=True)
    return embed_dict


def generate_negative_pairs(post_df= None, post_fn = None, N = 100):
    '''
    Given a DataFrame of positive pairs, generate N negative pairs.
    Negative pairs are pairs where phage and host are each present in the input dataframe, but the pair phage-host itself is not present.
    '''
    if post_df is None and post_fn is None:
        raise ValueError("Either post_df or post_fn should be provided.")
    if post_df is None:
        post_df = pd.read_csv(post_fn, sep='\t', header = 0, index_col = None)
    # Extract unique phages and hosts
    phages = post_df['Phage_ID'].unique()
    hosts = post_df['Host_ID'].unique()
    # Generate all possible pairs of phage and host
    all_possible_pairs = set(itertools.product(phages, hosts))
    # Convert positive pairs to a set of tuples
    positive_pairs = set(zip(post_df['Phage_ID'], post_df['Host_ID']))
    # Get negative pairs by subtracting positive pairs from all possible pairs
    negative_pairs = list(all_possible_pairs - positive_pairs)
    # Randomly sample N negative pairs
    if N > len(negative_pairs):
        N = len(negative_pairs)
    print(f"Number of possible negative pairs: {len(negative_pairs)}")
    sampled_negative_pairs = random.sample(negative_pairs, N)
    # Convert to DataFrame if desired
    negative_df = pd.DataFrame(sampled_negative_pairs, columns=['Phage_ID', 'Host_ID'])
    return negative_df

def filter_only_present_pairs(post_fn, phage_emb_dict=None, host_emb_dict=None):
    '''
    Given a DataFrame of pairs, filter out pairs where either phage or host is not present in the embeddings folder.
    '''
    post_df = pd.read_csv(post_fn, sep='\t', header = 0, index_col = None)
    phages = post_df['Phage_ID'].unique()
    hosts = post_df['Host_ID'].unique()
    if phage_emb_dict is not None:
        phages = phage_emb_dict.keys()
    if host_emb_dict is not None:
        hosts = host_emb_dict.keys()
    post_df = post_df[(post_df['Phage_ID'].isin(phages)) & (post_df['Host_ID'].isin(hosts))]
    return post_df

def prepare_training_data(phage_emb_dict, host_emb_dict, post_df, neg_df):
    '''
    the training data includes:
    - embeddings: for each pair of phage, host, (phgee_embed, host_embed) concatenated
    - labels: 1 for positive pairs, 0 for negative pairs
    '''
    embeddings = []
    post_df['label'] = 1
    neg_df['label'] = 0
    comb_df = pd.concat([post_df, neg_df], axis=0)
    comb_df.reset_index(drop=True, inplace=True)
    for i in range(comb_df.shape[0]):
        # import pdb; pdb.set_trace()
        phage_id = comb_df.iloc[i].Phage_ID
        host_id = comb_df.iloc[i].Host_ID
        phage_embed = phage_emb_dict[phage_id]
        host_embed = host_emb_dict[host_id]
        pair_embed = torch.cat((phage_embed, host_embed), dim=1)
        embeddings.append(pair_embed)
    labels = comb_df['label'].values
    embeddings = torch.cat(embeddings, axis=0)
    return embeddings, labels

if __name__ == '__main__':
    torch.manual_seed(123)
    np.random.seed(123)
    
    args = parse_arguments()

    phage_list =glob.glob(f'{args.phage_embed}/*.pt')
    host_list = glob.glob(f'{args.host_embed}/*.pt')
    phage_list = list(map(lambda x: x.split('/')[-1].split('.pt')[0], phage_list))
    host_list = list(map(lambda x: x.split('/')[-1].split('.pt')[0], host_list))

    phage_emb_dict = process_dict_embed(args.phage_embed)  # key: <phage_ID>, value: embedding tensor
    host_emb_dict = process_dict_embed(args.host_embed)     # key: <host_ID>, value: embedding tensor
    print('Done reading the embeddings')
    # Load positive pairs
    post_df = filter_only_present_pairs(args.train_positive_fn, phage_emb_dict, host_emb_dict)
    print('Done filtering the positive pairs')
    # Generate negative pairs
    if args.train_negative_fn is not None:
        neg_df = filter_only_present_pairs(args.train_negative_fn, phage_emb_dict, host_emb_dict)
    else:
        neg_df = generate_negative_pairs(post_df=post_df, N = len(post_df)*args.negative_fold)
    print('Done generating the negative pairs')
    # Prepare training data
    embeddings, labels = prepare_training_data(phage_emb_dict, host_emb_dict, post_df, neg_df)  # embeddings: (N, 8192), labels: (N,) where N is the number of pairs
    print('Done preparing the training data')
    scale_pos_weight = 1./np.mean(labels)
    print(f'scale_pos_weight is {scale_pos_weight}')
    # specify hyperparameters to be searched for
    param_grid = {
        'n_estimators': [50, 100, 200],
        'num_leaves': [31, 63, 127],
        'max_depth': [3, 5, 7, 10],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'learning_rate': [0.05, 0.1, 0.5],
        'scale_pos_weight': [scale_pos_weight/2, scale_pos_weight, scale_pos_weight*2],
    }

    # param_grid = {
    #     'n_estimators': [50],
    #     'num_leaves': [31],
    #     'max_depth': [3],
    #     'subsample': [0.8],
    #     'colsample_bytree': [0.8],
    #     'learning_rate': [0.05],
    #     'scale_pos_weight': [scale_pos_weight/2],
    # }
    bst = lgb.LGBMClassifier(objective='binary', boosting_type='gbdt')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
    # Perform RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=bst,
        param_distributions=param_grid,
        scoring=scorer,
        cv=cv,
        n_iter=10,
        verbose=0,
        n_jobs=5,  
        random_state=42
    )
    # fit model

    start_time = time()
    random_search.fit(embeddings, labels)
    bst = random_search.best_estimator_
    end_time = time()
    
    time_taken = end_time - start_time
    minutes = int(time_taken // 60)
    seconds = int(time_taken % 60)
    print(f"Time taken for RandomizedSearchCV: {minutes} minutes and {seconds} seconds")
    
    # save the model
    bst.booster_.save_model(args.save_fn)
    print(f"Model saved to {args.save_fn}")
    print("Done")
