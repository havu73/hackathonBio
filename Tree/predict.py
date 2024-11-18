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
import train

def parse_arguments():
    """Function to parse command line arguments."""
    parser = argparse.ArgumentParser(description="Make predictions based on the trained model")

    # Add arguments
    parser.add_argument('--trained_fn', type=str, help="path to the trained model")
    parser.add_argument('--phage_embed', type=str, default='phage_embeddings', help="path to phage embeddings folder")
    parser.add_argument('--host_embed', type=str, default='host_embeddings', help="path to host embeddings folder")
    parser.add_argument('--test_prompt', type=str, help="path to the file containing test prompts")
    parser.add_argument('--save_fn', type=str, required=True, help="path to save the model")
    # Parse the arguments and return them
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    phage_emb_dict = train.process_dict_embed(args.phage_embed)  # key: <phage_ID>, value: embedding tensor
    host_emb_dict = train.process_dict_embed(args.host_embed)  # key: <host_ID>, value: embedding tensor
    print('Done reading the embeddings')

    # Load the trained model
    bst = lgb.Booster(model_file=args.trained_fn)
    # Load the test prompts
    test_df = pd.read_csv(args.test_prompt, sep='\t', header=0, index_col=None)
    print('Done reading the test prompts')

    for idx, row in test_df.iterrows():
        phage_id = row['Phage_ID']
        host_id = row['Host_ID']
        phage_emb = phage_emb_dict[phage_id]
        host_emb = host_emb_dict[host_id]
        test_embedding = torch.cat((phage_emb, host_emb), dim=1)
        test_df.loc[idx, 'Prediction'] = bst.predict(test_embedding)[0]
    test_df.to_csv(args.save_fn, sep='\t', header=True, index=False)
    print('Done making predictions')

