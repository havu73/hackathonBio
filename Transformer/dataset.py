import torch
import os
import random
from torch.utils.data import Dataset
from glob import glob

random.seed(42)


class PairDataset(Dataset):
    def __init__(
        self, host_root, phage_root, pairs, labels, max_len, remove_non_exist=True
    ):
        super().__init__()
        self.host_root = host_root
        self.phage_root = phage_root
        self.pairs = pairs
        self.labels = labels
        self.max_len = max_len
        
        # only keep pairs and labels that have embeddings
        if remove_non_exist:
            valid_ids = [
                i for i, (host_idx, phage_idx) in enumerate(pairs) \
                    if os.path.exists(f'{host_root}/{host_idx}.pt') and os.path.exists(f'{phage_root}/{phage_idx}.pt')
            ]
        else:
            valid_ids = list(range(len(pairs)))
        self.pairs = [pairs[i] for i in valid_ids]
        self.labels = [labels[i] for i in valid_ids]
    
    def get_norm_constants(self):
        host_paths = glob(f'{self.host_root}/*.pt')
        host_embeddings = torch.cat([torch.load(path) for path in host_paths], dim=0)
        host_mean, host_std = host_embeddings.mean(dim=0, keepdim=True), host_embeddings.std(dim=0, keepdim=True)
        
        phage_paths = glob(f'{self.phage_root}/*.pt')
        phage_embeddings = torch.cat([torch.load(path) for path in phage_paths], dim=0)
        phage_mean, phage_std = phage_embeddings.mean(dim=0, keepdim=True), phage_embeddings.std(dim=0, keepdim=True)
        
        return host_mean, host_std, phage_mean, phage_std
    
    def set_norm_constants(self, host_mean, host_std, phage_mean, phage_std):
        self.host_mean = host_mean
        self.host_std = host_std
        self.phage_mean = phage_mean
        self.phage_std = phage_std
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        host_idx, phage_idx = self.pairs[idx]
        if not os.path.exists(f'{self.host_root}/{host_idx}.pt') or not os.path.exists(f'{self.phage_root}/{phage_idx}.pt'):
            return None, None, None, None
        
        host_embeddings = torch.load(f'{self.host_root}/{host_idx}.pt')  # N, 4096
        host_embeddings = (host_embeddings - self.host_mean) / self.host_std
        
        phage_embeddings = torch.load(f'{self.phage_root}/{phage_idx}.pt')  # M, 4096
        phage_embeddings = (phage_embeddings - self.phage_mean) / self.phage_std
        
        host_phage = torch.cat([host_embeddings, phage_embeddings], dim=0)

        # Padding
        pads = torch.zeros(self.max_len - host_phage.size(0), host_phage.size(1))
        host_phage = torch.cat([host_phage, pads], dim=0)

        return host_phage, torch.tensor([self.labels[idx]], dtype=torch.float32), host_embeddings.shape[0], phage_embeddings.shape[0]
