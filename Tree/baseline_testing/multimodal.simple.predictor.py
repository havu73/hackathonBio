#%%
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import umap
import numpy as np
import pandas as pd
import random
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
torch.manual_seed(123)
np.random.seed(123)

kle_phage_emb_path='/home/ec2-user/klebsiella_embeddings/phage_embeddings/'
kle_host_emb_path='/home/ec2-user/klebsiella_embeddings/host_embeddings/'

vibrio_phage_emb_path='/home/ec2-user/vibrio_embeddings/phage_embeddings/'
vibrio_host_emb_path='/home/ec2-user/vibrio_embeddings/host_embeddings/'

kle_phage_list = np.array(os.listdir(kle_phage_emb_path))
kle_host_list = np.array(os.listdir(kle_host_emb_path))

vibrio_phage_list = np.array(os.listdir(vibrio_phage_emb_path))
vibrio_host_list = np.array(os.listdir(vibrio_host_emb_path))

phage_list=np.concatenate((vibrio_phage_list,kle_phage_list))
host_list=np.concatenate((vibrio_host_list,kle_host_list))
    
# %%
### metadata:
kle_meta_df = pd.read_csv('/home/ec2-user/klebsiella_embeddings/study2_pairs_positive.txt',sep='\t')
kle_meta_df_tmp = pd.read_csv('/home/ec2-user/klebsiella_embeddings/study1_pairs_positive.txt',sep='\t')
kle_meta_df = pd.concat([kle_meta_df,kle_meta_df_tmp])
vibrio_meta_df = pd.read_csv('/home/ec2-user/vibrio_embeddings/pairs_positive.txt',sep='\t')

############## concatenate the summation embeddings  ##############
#%%

print(f'start collecting positive pairs...')
positive_pair_embeddings = []
indicators = []
for i in range(kle_meta_df.shape[0]):
    entry=kle_meta_df.iloc[i]
    phage_id = entry.phage
    bact_id = entry.bacterial_host
    try:
        evo_phage_embedding = torch.load(f'{kle_phage_emb_path}/{phage_id}.pt').sum(dim=0,keepdim=True)
        evo_bact_embedding = torch.load(f'{kle_host_emb_path}/{bact_id}.pt').sum(dim=0,keepdim=True)
        positive_pair_embedding = torch.cat((evo_phage_embedding,evo_bact_embedding),dim=1)
        positive_pair_embeddings.append(positive_pair_embedding)
        indicators.append(True)
    except:
        continue
print(f'finish collecting klebsiella, with {len(positive_pair_embeddings)} positive embeddings in total')

for i in range(vibrio_meta_df.shape[0]):
    entry=vibrio_meta_df.iloc[i]
    phage_id = entry.phage_id
    bact_id = entry.bacterial_id
    try:
        evo_phage_embedding = torch.load(f'{vibrio_phage_emb_path}/{phage_id}.pt').sum(dim=0,keepdim=True)
        evo_bact_embedding = torch.load(f'{vibrio_host_emb_path}/{bact_id}.pt').sum(dim=0,keepdim=True)
        positive_pair_embedding = torch.cat((evo_phage_embedding,evo_bact_embedding),dim=1)
        positive_pair_embeddings.append(positive_pair_embedding)
        indicators.append(True)
    except:
        continue

print(f'finish collecting vibrio, with {len(positive_pair_embeddings)} positive embeddings in total')
positive_pair_embeddings = torch.cat(positive_pair_embeddings,axis=0)# %%
random_pair_embeddings = []
for i in range(positive_pair_embeddings.shape[0]):
    random_phage_int = random.randint(0, len(phage_list)-1)
    random_bact_int = random.randint(0, len(host_list)-1)
    phage_id = phage_list[random_phage_int]
    bact_id = host_list[random_bact_int]
    try:
        evo_phage_embedding = torch.load(f'{kle_phage_emb_path}/{phage_id}').mean(dim=0,keepdim=True)
    except:
        evo_phage_embedding = torch.load(f'{vibrio_phage_emb_path}/{phage_id}').mean(dim=0,keepdim=True)
    try:
        evo_bact_embedding = torch.load(f'{kle_host_emb_path}/{bact_id}').mean(dim=0,keepdim=True)
    except:
        evo_bact_embedding = torch.load(f'{vibrio_host_emb_path}/{bact_id}').mean(dim=0,keepdim=True)
        
    random_pair_embedding = torch.cat((evo_phage_embedding,evo_bact_embedding),dim=1)
    random_pair_embeddings.append(random_pair_embedding)
    indicators.append(False)
    
random_pair_embeddings = torch.cat(random_pair_embeddings,axis=0)
print(f'finish generating random pairs as negative control, with {len(random_pair_embeddings)} negative embeddings in total')

pc_ncomp=50
n_neighbors=50
combined_pair_embeddings = torch.cat((positive_pair_embeddings,random_pair_embeddings),axis=0)
indicators=np.array(indicators)
combined_pair_embeddings = StandardScaler().fit_transform(combined_pair_embeddings)
pca = PCA(n_components=pc_ncomp, svd_solver='arpack')
pca_embedding=pca.fit_transform(combined_pair_embeddings)
print(f'Finish information contraction using PCA')

# fumap=umap.UMAP(n_neighbors=n_neighbors)
# umap_embedding = fumap.fit_transform(pca_embedding)
# plt.scatter(umap_embedding[indicators,0],umap_embedding[indicators,1],c='r',label='positive',alpha=0.1)
# plt.scatter(umap_embedding[~indicators,0],umap_embedding[~indicators,1],c='b',label='random',alpha=0.1)
# plt.title(f'{pc_ncomp} PCs \nneighbor{n_neighbors} UMAP ')
# plt.legend()



# %%
## T-test
positive_1pc=pca_embedding[indicators,0]
random_1pc=pca_embedding[~indicators,0]
plt.hist(positive_1pc[positive_1pc>-100],alpha=0.5,label='positive')
plt.hist(random_1pc[random_1pc>-100],alpha=0.5,label='random')
plt.legend()
plt.show()
############## concatenate the embeddings  ##############
# %%
############## add the embeddings  ##############

print(f'start collecting positive pairs...')
positive_pair_embeddings = []
indicators = []
for i in range(kle_meta_df.shape[0]):
    entry=kle_meta_df.iloc[i]
    phage_id = entry.phage
    bact_id = entry.bacterial_host
    try:
        evo_phage_embedding = torch.load(f'{kle_phage_emb_path}/{phage_id}.pt').sum(dim=0,keepdim=True)
        evo_bact_embedding = torch.load(f'{kle_host_emb_path}/{bact_id}.pt').sum(dim=0,keepdim=True)
        positive_pair_embedding = evo_phage_embedding+evo_bact_embedding
        positive_pair_embeddings.append(positive_pair_embedding)
        indicators.append(True)
    except:
        continue
    
print(f'finish collecting klebsiella, with {len(positive_pair_embeddings)} positive embeddings in total')

for i in range(vibrio_meta_df.shape[0]):
    entry=vibrio_meta_df.iloc[i]
    phage_id = entry.phage_id
    bact_id = entry.bacterial_id
    try:
        evo_phage_embedding = torch.load(f'{vibrio_phage_emb_path}/{phage_id}.pt').sum(dim=0,keepdim=True)
        evo_bact_embedding = torch.load(f'{vibrio_host_emb_path}/{bact_id}.pt').sum(dim=0,keepdim=True)
        positive_pair_embedding = evo_phage_embedding+evo_bact_embedding
        positive_pair_embeddings.append(positive_pair_embedding)
        indicators.append(True)
    except:
        continue

print(f'finish collecting vibrio, with {len(positive_pair_embeddings)} positive embeddings in total')

positive_pair_embeddings = torch.cat(positive_pair_embeddings,axis=0)# %%
random_pair_embeddings = []
for i in range(positive_pair_embeddings.shape[0]):
    random_phage_int = random.randint(0, len(phage_list)-1)
    random_bact_int = random.randint(0, len(host_list)-1)
    phage_id = phage_list[random_phage_int]
    bact_id = host_list[random_bact_int]
    try:
        evo_phage_embedding = torch.load(f'{kle_phage_emb_path}/{phage_id}').mean(dim=0,keepdim=True)
    except:
        evo_phage_embedding = torch.load(f'{vibrio_phage_emb_path}/{phage_id}').mean(dim=0,keepdim=True)
    try:
        evo_bact_embedding = torch.load(f'{kle_host_emb_path}/{bact_id}').mean(dim=0,keepdim=True)
    except:
        evo_bact_embedding = torch.load(f'{vibrio_host_emb_path}/{bact_id}').mean(dim=0,keepdim=True)
        
    random_pair_embedding = evo_phage_embedding+evo_bact_embedding
    random_pair_embeddings.append(random_pair_embedding)
    indicators.append(False)

print(f'finish generating random pairs as negative control, with {len(random_pair_embeddings)} negative embeddings in total')
random_pair_embeddings = torch.cat(random_pair_embeddings,axis=0)
combined_pair_embeddings = torch.cat((positive_pair_embeddings,random_pair_embeddings),axis=0)
indicators=np.array(indicators)

pc_ncomp=50

combined_pair_embeddings = StandardScaler().fit_transform(combined_pair_embeddings)
pca = PCA(n_components=pc_ncomp, svd_solver='arpack')
pca_embedding=pca.fit_transform(combined_pair_embeddings)
print(f'Finish information contraction using PCA')

# %%
## supervised learning

bst = XGBClassifier(n_estimators=10, max_depth=5, learning_rate=1, objective='binary:logistic')
# fit model
X_train, X_test, y_train, y_test = train_test_split(pca_embedding,indicators)
print(f'Split the data into {X_train.shape[0]} train set and {X_test.shape[0]} test set')


bst.fit(X_train, y_train)
print(f'Finish training the model')

preds = bst.predict(X_test)

score = accuracy_score(y_test,preds)
print(f'accuracy score on test set: {score}')

