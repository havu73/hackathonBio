# %%
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, average_precision_score, roc_auc_score
import matplotlib.pyplot as plt
from time import time
import os
import umap
import numpy as np
import pandas as pd
import random
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,log_loss
import lightgbm as lgb
import argparse


def parse_arguments():
    """Function to parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simple calculator using argparse")
    
    # Add arguments
    parser.add_argument('--negative_fold', type=int, default=10, help="positive/negative pair ratio")
    parser.add_argument('--data', type=str, help="input")
    parser.add_argument('--embeddingPath', type=str, default='/mnt/efs/fs1/data/embeddings')
    parser.add_argument('--metaPath', type=str, default='/mnt/efs/fs1/data/metadata')

    # Parse the arguments and return them
    return parser.parse_args()

if __name__ == '__main__':
    torch.manual_seed(123)
    np.random.seed(123)
    
    args = parse_arguments()

    embeddingPath=args.embeddingPath
    metaPath=args.metaPath
    negative_fold=args.negative_fold
    # datanames = ['vibrio']
    # datanames = [args.data]
    dataname = args.data
    phage_list = None
    host_list = None
    
    test_prompt=pd.read_csv(f"{metaPath}/{dataname}/test_prompt.txt",sep="\t")

    exec(f'{dataname}_phage_emb_path="{embeddingPath}/{dataname}_embeddings/phage_embeddings/"')
    exec(f'{dataname}_host_emb_path="{embeddingPath}/{dataname}_embeddings/host_embeddings/"')
    
    exec(f'{dataname}_phage_list=os.listdir({dataname}_phage_emb_path)')
    exec(f'{dataname}_host_list=os.listdir({dataname}_host_emb_path)')
    
    exec(f'{dataname}_meta_df=pd.read_csv("{metaPath}/{dataname}/train_positive_pairs.txt",sep="\t")')
    
    if phage_list is None:
        phage_list=eval(f'{dataname}_phage_list')
        host_list=eval(f'{dataname}_host_list')
    else:
        phage_list = phage_list.extend(eval(f'{dataname}_phage_list'))
        host_list = host_list.extend(eval(f'{dataname}_host_list'))

    ############## concatenate the summation embeddings  ##############
    #%%

    print(f'start collecting positive pairs...')
    positive_pair_embeddings = []
    indicators = []
  
    missing=0
# for dataname in ['phageDB']:
    for i in range(eval(f'{dataname}_meta_df').shape[0]):
        entry=eval(f'{dataname}_meta_df').iloc[i]
        phage_id = entry.Phage_ID
        if dataname=='phageDB':
            phage_id=f'phageDB{phage_id}'
        bact_id = entry.Host_ID
        try:
            phage_emb_path=eval(f'{dataname}_phage_emb_path')
            host_emb_path=eval(f'{dataname}_host_emb_path')
            evo_phage_embedding = torch.load(f'{phage_emb_path}/{phage_id}.pt').sum(dim=0,keepdim=True)
            evo_host_embedding = torch.load(f'{host_emb_path}/{bact_id}.pt').sum(dim=0,keepdim=True)
            positive_pair_embedding = torch.cat((evo_phage_embedding,evo_host_embedding),dim=1)
            positive_pair_embeddings.append(positive_pair_embedding)
            indicators.append(True)
        except Exception as e:
            missing += 1
            # print(e)
            continue
    print(f'finish collecting {dataname}, with {len(positive_pair_embeddings)} positive embeddings in total, miss {missing}')

    positive_pair_embeddings = torch.cat(positive_pair_embeddings,axis=0)# %%
    random_pair_embeddings = []

    Npos=positive_pair_embeddings.shape[0]

    missing=0
    
    if Npos > 1000:
        negative_fold = 5
    # print(test_prompt)
    for i in range(Npos*negative_fold):
        ##### if use meta train pair only #####
        # random_phage_int = random.randint(0, Npos-1)
        # random_bact_int = random.randint(0, Npos-1)
        # if random_phage_int == random_bact_int:
        #     ## false negative
        #     continue
        
        # phage_id = eval(f'{dataname}_meta_df').iloc[random_phage_int].Phage_ID
        # bact_id = eval(f'{dataname}_meta_df').iloc[random_bact_int].Host_ID
        # phage_id = f'{phage_id}.pt'
        # if dataname=='phageDB':
        #     phage_id=f'phageDB{phage_id}'
        # bact_id = f'{bact_id}.pt'
        ##### if use meta train pair only #####
        
        ##### if use all random pairs #####
        random_phage_int = random.randint(0, len(phage_list)-1)
        random_bact_int = random.randint(0, len(host_list)-1)
        
        phage_id = phage_list[random_phage_int]
        bact_id = host_list[random_bact_int]
        # print(f'phage_id: {phage_id[:-2]} -- bact_id: {bact_id[:-2]}')
        
        condition =((test_prompt['Phage_ID']==phage_id[:-3])&(test_prompt['Host_ID']==bact_id[:-3]))
        if np.sum(condition)>0: ### present in testing data
            continue
        # phage_id = eval(f'{dataname}_meta_df').iloc[random_phage_int].Phage_ID
        # bact_id = eval(f'{dataname}_meta_df').iloc[random_bact_int].Host_ID
        ##### if use all random pairs #####
        
        
        phage_emb_path=eval(f'{dataname}_phage_emb_path')
        host_emb_path=eval(f'{dataname}_host_emb_path')
        try:
            evo_phage_embedding = torch.load(f'{phage_emb_path}/{phage_id}').sum(dim=0,keepdim=True)
            evo_host_embedding = torch.load(f'{host_emb_path}/{bact_id}').sum(dim=0,keepdim=True)
        except Exception as e:
            # print(e)
            # print(f'error loading {phage_emb_path}/{phage_id}')
            missing += 1
            continue
    
        random_pair_embedding = torch.cat((evo_phage_embedding,evo_host_embedding),dim=1)
        random_pair_embeddings.append(random_pair_embedding)
        indicators.append(False)
        
    random_pair_embeddings = torch.cat(random_pair_embeddings,axis=0)
    print(f'finish generating random pairs as negative control, with {len(random_pair_embeddings)} negative embeddings in total, miss {missing}')

    pc_ncomp=100
    n_neighbors=50

    ## How many negative pairs use for training purpose
    train_neg_portion=positive_pair_embeddings.shape[0]


    combined_pair_embeddings = torch.cat((positive_pair_embeddings,random_pair_embeddings),axis=0)

    indicators=np.array(indicators)
    # indicators_train = indicators[:2*train_neg_portion]
    # assert combined_pair_embeddings_train.shape[0]==indicators_train.shape[0],f'X train shape: {combined_pair_embeddings_train.shape}; y train shape: {indicators_train.shape}'


    scale_pos_weight = 1./np.mean(indicators)
    print(f'scale_pos_weight is {scale_pos_weight}')
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
    #     'n_estimators': [100],
    #     'max_depth': [5],
    #     'learning_rate': [0.5],
    #     'scale_pos_weight': [scale_pos_weight],
    #     'gamma': [0]
    # }



    
    
    
    # bst = XGBClassifier(objective='binary:logistic',scale_pos_weight=scale_pos_weight,n_estimators=100,max_depth=5)
    # bst.fit(combined_pair_embeddings, indicators)
                        # tree_method='hist',
                        # device='cuda' )
                        
                        
    # bst = XGBClassifier(objective='binary:logistic')
    bst = lgb.LGBMClassifier(objective='binary', boosting_type='gbdt')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
    # scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)
    # Perform RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=bst,
        param_distributions=param_grid,
        scoring=scorer,
        cv=cv,
        n_iter=10,  # Number of hyperparameter combinations to try
        verbose=0,
        n_jobs=5,  
        random_state=42
    )
    # fit model

    start_time = time()
    random_search.fit(combined_pair_embeddings, indicators)
    bst = random_search.best_estimator_
    end_time = time()
    
    time_taken = end_time - start_time
    minutes = int(time_taken // 60)
    seconds = int(time_taken % 60)
    print(f"Time taken for RandomizedSearchCV: {minutes} minutes and {seconds} seconds")
    

    #### Testing part #####
    
    test_embeddings = []
    prediction = np.full(test_prompt.shape[0], np.nan)
    missing=0
    for i in range(test_prompt.shape[0]):
        entry=test_prompt.iloc[i]
        phage_id = entry.Phage_ID
        if dataname=='phageDB':
            phage_id=f'phageDB{phage_id}'
        bact_id = entry.Host_ID
        
        phage_emb_path = globals().get(f'{dataname}_phage_emb_path', None)
        host_emb_path = globals().get(f'{dataname}_host_emb_path', None)
        # phage_emb_path=eval(f'{dataname}_phage_emb_path')
        # host_emb_path=eval(f'{dataname}_host_emb_path')
        try:
            evo_phage_embedding = torch.load(f'{phage_emb_path}/{phage_id}.pt').sum(dim=0,keepdim=True)
            evo_host_embedding = torch.load(f'{host_emb_path}/{bact_id}.pt').sum(dim=0,keepdim=True)
        except:
            missing += 1
            continue
        
        test_embedding = torch.cat((evo_phage_embedding,evo_host_embedding),dim=1)
        prediction[i]=bst.predict_proba(test_embedding)[0, 1] 
        # test_embeddings.append(test_embedding)
        
    # test_embeddings = torch.cat(test_embeddings,axis=0)
    
    # preds_proba = bst.predict_proba(test_embeddings)[:, 1] 
    print(f'testing data miss {missing} embeddings')
    test_prompt['prediction']=prediction
    test_prompt.to_csv(f'/home/ec2-user/users/boyang/explore/lightgbm/predict/{dataname}.boyang.predict',sep='\t',index=None)

