import torch
import os
import random
from torch.utils.data import Dataset
import glob
import pandas as pd
from typing import List, Iterator
from torch.utils.data import DataLoader, Sampler
random.seed(42)


def process_dict_embed(embed_folder):
    '''
    Given a folder of embeddings, <phage/bacteria_ID>.pt files, return a dictionary of embeddings
    key: <phage/bacteria_ID>,
    value: embedding tensor
    '''
    embed_dict = {}
    for embed_fn in glob.glob(f'{embed_folder}/*.pt'):
        embed_id = embed_fn.split('/')[-1].split('.pt')[0]
        embed_dict[embed_id] = torch.load(f'{embed_fn}')
    return embed_dict


class PairDataset(Dataset):
    def __init__(
        self, host_root, phage_root, pairs, labels, max_len):
        super().__init__()
        self.host_root = host_root
        self.phage_root = phage_root
        self.host_embed_dict = process_dict_embed(host_root)
        self.phage_embed_dict = process_dict_embed(phage_root)
        self.pairs = pairs
        self.labels = labels
        self.max_len = max_len
        self.normalized = False

    
    def get_norm_constants(self):
        host_paths = glob.glob(f'{self.host_root}/*.pt')
        host_embeddings = torch.cat([torch.load(path) for path in host_paths], dim=0)
        host_mean, host_std = host_embeddings.mean(dim=0, keepdim=True), host_embeddings.std(dim=0, keepdim=True)
        phage_paths = glob.glob(f'{self.phage_root}/*.pt')
        phage_embeddings = torch.cat([torch.load(path) for path in phage_paths], dim=0)
        phage_mean, phage_std = phage_embeddings.mean(dim=0, keepdim=True), phage_embeddings.std(dim=0, keepdim=True)
        self.host_mean = host_mean
        self.host_std = host_std
        self.phage_mean = phage_mean
        self.phage_std = phage_std
        return host_mean, host_std, phage_mean, phage_std

    def set_norm_constants(self, host_mean, host_std, phage_mean, phage_std):
        '''
        This set function is necessary because we will set the mean and std of the embeddings based on the training set
        '''
        self.host_mean = host_mean
        self.host_std = host_std
        self.phage_mean = phage_mean
        self.phage_std = phage_std
        return

    def normalize_data(self):
        if not hasattr(self, 'host_mean'):
            self.get_norm_constants()
        for key in self.host_embed_dict:
            self.host_embed_dict[key] = (self.host_embed_dict[key] - self.host_mean) / self.host_std
        for key in self.phage_embed_dict:
            self.phage_embed_dict[key] = (self.phage_embed_dict[key] - self.phage_mean) / self.phage_std
        self.normalized = True
        return
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        if not self.normalized:
            self.normalize_data()
        host_idx, phage_idx = self.pairs[idx]
        host_embeddings = self.host_embed_dict[host_idx]  # N, 4096
        phage_embeddings = self.phage_embed_dict[phage_idx]  # M, 4096
        
        host_phage = torch.cat([host_embeddings, phage_embeddings], dim=0)

        # Padding
        pads = torch.zeros(self.max_len - host_phage.size(0), host_phage.size(1))
        host_phage = torch.cat([host_phage, pads], dim=0)

        return host_phage, torch.tensor([self.labels[idx]], dtype=torch.float32), host_embeddings.shape[0], phage_embeddings.shape[0]


# Custom sampler to ensure equal number of positive and negative examples in each batch
class BalancedBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.positive_indices = [i for i, label in enumerate(dataset.labels) if label == 1]
        self.negative_indices = [i for i, label in enumerate(dataset.labels) if label == 0]
        self.smallest_class = min(len(self.positive_indices), len(self.negative_indices))

    def __iter__(self) -> Iterator[List[int]]:
        positive_indices = self.positive_indices.copy()
        negative_indices = self.negative_indices.copy()
        random.shuffle(positive_indices)
        random.shuffle(negative_indices)

        # Ensure we have equal numbers of positive and negative samples
        positive_indices = positive_indices[:self.smallest_class]
        negative_indices = negative_indices[:self.smallest_class]

        # Combine and shuffle
        all_indices = positive_indices + negative_indices
        random.shuffle(all_indices)

        # Yield batches
        for i in range(0, len(all_indices), self.batch_size):
            yield all_indices[i:i + self.batch_size]

    def __len__(self):
        return (self.smallest_class * 2 + self.batch_size - 1) // self.batch_size


def create_dataloaders(args):
    positive_pairs_df = pd.read_csv(args.train_positive_fn, sep='\t')
    negative_pairs_df = pd.read_csv(args.train_negative_fn, sep='\t')
    # get pairs
    positive_pairs = [(row['Host_ID'], row['Phage_ID']) for _, row in positive_pairs_df.iterrows()]
    negative_pairs = [(row['Host_ID'], row['Phage_ID']) for _, row in negative_pairs_df.iterrows()]
    # shuffle pairs, just to be safe
    random.shuffle(positive_pairs)
    random.shuffle(negative_pairs)

    # Prepare validation set with 1:1 ratio
    val_size = int((1 - args.train_ratio) * len(positive_pairs_df))
    val_positive = positive_pairs[:val_size]
    val_negative = negative_pairs[:val_size]
    val_pairs = val_positive + val_negative
    val_labels = [1] * len(val_positive) + [0] * len(val_negative)

    # Prepare training set
    train_positive = positive_pairs[val_size:]
    if args.negative_pairs_ratio == -1:
        train_negative = negative_pairs[val_size:]
    else:
        train_negative = negative_pairs[val_size:val_size + len(train_positive) * args.negative_pairs_ratio]

    train_pairs = train_positive + train_negative
    train_labels = [1] * len(train_positive) + [0] * len(train_negative)

    # Create datasets
    train_dataset = PairDataset(
        host_root=args.host_embed,
        phage_root=args.phage_embed,
        pairs=train_pairs,
        labels=train_labels,
        max_len=args.max_len
    )
    norm_constants = train_dataset.get_norm_constants()
    train_dataset.set_norm_constants(*norm_constants)

    # Save norm_constants to disk
    torch.save(norm_constants, os.path.join(args.root_dir, 'norm_constants.pth'))

    val_dataset = PairDataset(
        host_root=args.host_embed,
        phage_root=args.phage_embed,
        pairs=val_pairs,
        labels=val_labels,
        max_len=args.max_len
    )
    val_dataset.set_norm_constants(*norm_constants)  # norm constant was calculated from training data

    # normalize data
    train_dataset.normalize_data()
    val_dataset.normalize_data()

    train_sampler = BalancedBatchSampler(train_dataset, args.batch_size)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader