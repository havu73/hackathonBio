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

torch.manual_seed(123)
np.random.seed(123)

phage_emb_path='/home/ec2-user/vibrio_data/embeddings/phage_embeddings/'
host_emb_path='/home/ec2-user/vibrio_data/embeddings/host_embeddings/'
phage_list = os.listdir(phage_emb_path)
host_list = os.listdir(host_emb_path)

# %%
### metadata:
meta_df = pd.read_csv('/home/ec2-user/vibrio_data/pairs_positive.txt',sep='\t')

# %%
############## add the embeddings -- inter-species summation + concatenation  ##############
positive_pair_embeddings = []
indicators = []
for i in range(meta_df.shape[0]):
    entry=meta_df.iloc[i]
    phage_id = entry.phage_id
    bact_id = entry.bacterial_id
    try:
        evo_phage_embedding = torch.load(f'{phage_emb_path}/{phage_id}.pt').sum(dim=0,keepdim=True)
        evo_bact_embedding = torch.load(f'{host_emb_path}/{bact_id}.pt').sum(dim=0,keepdim=True)
        positive_pair_embedding = evo_phage_embedding+evo_bact_embedding
        positive_pair_embeddings.append(positive_pair_embedding)
        indicators.append(True)
    except:
        continue
positive_pair_embeddings = torch.cat(positive_pair_embeddings,axis=0)# %%
random_pair_embeddings = []
for i in range(positive_pair_embeddings.shape[0]):
    random_phage_int = random.randint(0, len(phage_list)-1)
    random_bact_int = random.randint(0, len(host_list)-1)
    phage_id = phage_list[random_phage_int]
    bact_id = host_list[random_bact_int]
    evo_phage_embedding = torch.load(f'{phage_emb_path}/{phage_id}').mean(dim=0,keepdim=True)
    evo_bact_embedding = torch.load(f'{host_emb_path}/{bact_id}').mean(dim=0,keepdim=True)
    random_pair_embedding=evo_phage_embedding+evo_bact_embedding
    random_pair_embeddings.append(random_pair_embedding)
    indicators.append(False)
random_pair_embeddings = torch.cat(random_pair_embeddings,axis=0)
# %%
pc_ncomp=50
n_neighbors=50

combined_pair_embeddings = torch.cat((positive_pair_embeddings,random_pair_embeddings),axis=0)
indicators=np.array(indicators)
combined_pair_embeddings = StandardScaler().fit_transform(combined_pair_embeddings)
pca = PCA(n_components=pc_ncomp, svd_solver='arpack')
pca_embedding=pca.fit_transform(combined_pair_embeddings)
fumap=umap.UMAP(n_neighbors=n_neighbors)
umap_embedding = fumap.fit_transform(pca_embedding)
plt.scatter(umap_embedding[indicators,0],umap_embedding[indicators,1],c='r',label='positive',alpha=0.1)
plt.scatter(umap_embedding[~indicators,0],umap_embedding[~indicators,1],c='b',label='random',alpha=0.1)
plt.title(f'{pc_ncomp} PCs \nneighbor{n_neighbors} UMAP ')
plt.legend()

# %%
