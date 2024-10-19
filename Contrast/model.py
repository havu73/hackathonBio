import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class PhageBactPairDataset(Dataset):
    def __init__(self, pair_df, method, embedding_dir):
        self.pairs = pair_df[['Host_ID', 'Phage_ID']]
        self.labels = pair_df['Label'].values
        self.method = method
        self.embedding_dir = embedding_dir
        
    def __len__(self):
        return len(self.labels)
    
    def normalize_0_1(self, tensor):
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())
    
    def __getitem__(self, idx):
        pair = self.pairs.iloc[idx]
        label = self.labels[idx]
        
        if self.method == 'sum':
            pair_phage = torch.sum(torch.load(self.embedding_dir + '/phage_embeddings/' + pair['Phage_ID'] + '.pt'), 0)
            pair_phage = self.normalize_0_1(pair_phage)
            pair_bact = torch.sum(torch.load(self.embedding_dir + '/host_embeddings/' + pair['Host_ID'] + '.pt'), 0)
            pair_bact = self.normalize_0_1(pair_bact)
            
            pair_phage = torch.reshape(pair_phage, (1, pair_phage.size()[0]))
            pair_bact = torch.reshape(pair_bact, (1, pair_bact.size()[0]))

        return pair_phage, pair_bact, torch.tensor(label)

class CNNEmbedding(nn.Module):
    def __init__(self):
        super(CNNEmbedding, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, (1,3), stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, (1,3), stride=1)
        self.bn2 = nn.BatchNorm2d(128) 
        self.fc1 = nn.Linear(128*1022, 1024)
        self.fc2 = nn.Linear(1024, 512)
        
    def forward(self, x):
        x = self.bn1(F.max_pool2d(F.relu(self.conv1(x)), (1,2)))
        x = self.bn2(F.max_pool2d(F.relu(self.conv2(x)), (1,2)))
        return self.fc2(F.relu(self.fc1(torch.flatten(x, 1))))
    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, phage, bact, label):
        p_h_distance = phage - bact
        dist_norm_squared = torch.sum(torch.pow(p_h_distance, 2), 1)
        dist_norm = torch.sqrt(dist_norm_squared)
        neg = torch.pow(torch.clamp(self.margin - dist_norm, min=0), 2)
        loss_per_batch = label*dist_norm_squared + (1 - label)*neg
        return 0.5*torch.mean(loss_per_batch)




