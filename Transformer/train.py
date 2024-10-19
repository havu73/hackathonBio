import os
import random
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import wandb
from tqdm import tqdm
from dataset import PairDataset
from model import TransformerClassifier
from lr_scheduler import LinearWarmupCosineAnnealingLR
from typing import List, Iterator

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

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
    positive_pairs_df = pd.read_csv(args.positive_pairs_file, sep='\t')
    positive_pairs = [(row['Host_ID'], row['Phage_ID']) for _, row in positive_pairs_df.iterrows()]
    
    host_ids = positive_pairs_df['Host_ID'].unique()
    phage_ids = positive_pairs_df['Phage_ID'].unique()
    negative_pairs = [
        (host_id, phage_id) 
        for host_id in host_ids 
        for phage_id in phage_ids if (host_id, phage_id) not in positive_pairs
    ]

    random.shuffle(positive_pairs)
    random.shuffle(negative_pairs)

    # Prepare validation set with 1:1 ratio
    val_size = int((1 - args.train_ratio) * len(positive_pairs))
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
        host_root=args.host_embeddings_path,
        phage_root=args.phage_embeddings_path,
        pairs=train_pairs,
        labels=train_labels,
        max_len=args.max_len
    )
    norm_constants = train_dataset.get_norm_constants()
    train_dataset.set_norm_constants(*norm_constants)
    
    # Save norm_constants to disk
    torch.save(norm_constants, os.path.join(args.root_dir, 'norm_constants.pth'))
    
    val_dataset = PairDataset(
        host_root=args.host_embeddings_path,
        phage_root=args.phage_embeddings_path,
        pairs=val_pairs,
        labels=val_labels,
        max_len=args.max_len
    )
    val_dataset.set_norm_constants(*norm_constants)

    train_sampler = BalancedBatchSampler(train_dataset, args.batch_size)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader

def get_metrics(all_labels, all_preds):
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    # ROC AUC score
    roc_auc = roc_auc_score(all_labels, all_preds)

    return accuracy, precision, recall, f1, roc_auc

def train_epoch(model, loader, criterion, optimizer, lr_scheduler, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for sequences, labels, host_len, phage_len in tqdm(loader, desc="Training"):
        sequences, labels, host_len, phage_len = sequences.to(device), labels.to(device), host_len.to(device), phage_len.to(device)

        optimizer.zero_grad()
        outputs = model(sequences, host_len, phage_len)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(outputs.cpu().detach().numpy() > 0.5)
        all_labels.extend(labels.cpu().numpy())
    
    lr_scheduler.step()

    avg_loss = total_loss / len(loader)
    accuracy, precision, recall, f1, roc_auc = get_metrics(all_labels, all_preds)

    return avg_loss, accuracy, precision, recall, f1, roc_auc

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels, host_len, phage_len in tqdm(loader, desc="Validating"):
            sequences, labels, host_len, phage_len = sequences.to(device), labels.to(device), host_len.to(device), phage_len.to(device)

            outputs = model(sequences, host_len, phage_len)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy() > 0.5)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy, precision, recall, f1, roc_auc = get_metrics(all_labels, all_preds)

    return avg_loss, accuracy, precision, recall, f1, roc_auc

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.root_dir = os.path.join(args.root_dir, args.dataset, args.exp_name)
    args.checkpoint_dir = os.path.join(args.root_dir, "checkpoints")
    
    os.makedirs(args.root_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    assert args.dataset in ["klebsiella", "vibrio", "ecoli", "vikecoli", "phageDB", "phageScope"]
    
    if args.dataset == "klebsiella":
        args.positive_pairs_file = "/mnt/efs/fs1/data/metadata/klebsiella/train_positive_pairs.txt"
        args.host_embeddings_path = "/mnt/efs/fs1/data/embeddings/klebsiella_embeddings/host_embeddings"
        args.phage_embeddings_path = "/mnt/efs/fs1/data/embeddings/klebsiella_embeddings/phage_embeddings"
    elif args.dataset == "vibrio":
        args.positive_pairs_file = "/mnt/efs/fs1/data/metadata/vibrio/train_positive_pairs.txt"
        args.host_embeddings_path = "/mnt/efs/fs1/data/embeddings/vibrio_embeddings/host_embeddings"
        args.phage_embeddings_path = "/mnt/efs/fs1/data/embeddings/vibrio_embeddings/phage_embeddings"
    elif args.dataset == "ecoli":
        args.positive_pairs_file = "/mnt/efs/fs1/data/metadata/ecoli/train_positive_pairs.txt"
        args.host_embeddings_path = "/mnt/efs/fs1/data/embeddings/ecoli_embeddings/host_embeddings"
        args.phage_embeddings_path = "/mnt/efs/fs1/data/embeddings/ecoli_embeddings/phage_embeddings"
    elif args.dataset == "vikecoli":
        args.positive_pairs_file = "/mnt/efs/fs1/data/metadata/vikecoli/train_positive_pairs.txt"
        args.host_embeddings_path = "/mnt/efs/fs1/data/embeddings/vikecoli_embeddings/host_embeddings"
        args.phage_embeddings_path = "/mnt/efs/fs1/data/embeddings/vikecoli_embeddings/phage_embeddings"
    elif args.dataset == "phageDB":
        args.positive_pairs_file = "/mnt/efs/fs1/data/metadata/phageDB/train_positive_pairs.txt"
        args.host_embeddings_path = "/mnt/efs/fs1/data/embeddings/phageDB_embeddings/host_embeddings"
        args.phage_embeddings_path = "/mnt/efs/fs1/data/embeddings/phageDB_embeddings/phage_embeddings"
    elif args.dataset == "phageScope":
        args.positive_pairs_file = "/mnt/efs/fs1/data/metadata/phageScope/train_positive_pairs.txt"
        args.host_embeddings_path = "/mnt/efs/fs1/data/embeddings/phageScope_embeddings/host_embeddings"
        args.phage_embeddings_path = "/mnt/efs/fs1/data/embeddings/phageScope_embeddings/phage_embeddings"

    # Initialize wandb
    wandb.init(project=args.wandb_project, config=args, name=args.exp_name, dir=args.root_dir)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(args)

    # Initialize model
    model = TransformerClassifier(
        input_dim=args.input_dim,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_len=args.max_len
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    lr_scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=args.scheduler_warmup_epochs,
        max_epochs=args.scheduler_max_epochs
    )

    # Load checkpoint if resuming training
    start_epoch = 0
    best_val_f1 = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_f1 = checkpoint['best_val_f1']
        print(f"Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        train_loss, train_acc, train_prec, train_rec, train_f1, train_roc_auc = train_epoch(model, train_loader, criterion, optimizer, lr_scheduler, device)
        val_loss, val_acc, val_prec, val_rec, val_f1, val_roc_auc = validate(model, val_loader, criterion, device)

        # Log metrics
        wandb.log({
            "epoch": epoch+1,
            "lr": optimizer.param_groups[0]['lr'],
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "train/precision": train_prec,
            "train/recall": train_rec,
            "train/f1": train_f1,
            "train/roc_auc": train_roc_auc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "val/precision": val_prec,
            "val/recall": val_rec,
            "val/f1": val_f1,
            "val/roc_auc": val_roc_auc,
        })

        print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

        # Save the last checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_f1': best_val_f1,
        }, os.path.join(args.checkpoint_dir, f'epoch_{epoch+1}.pth'))

        # Save the best checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_f1': best_val_f1,
            }, os.path.join(args.checkpoint_dir, 'best_checkpoint.pth'))
            print("Saved new best checkpoint")

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer Classifier")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--negative_pairs_ratio", type=int, default=-1, help="Ratio of negative to positive pairs")
    parser.add_argument("--train_ratio", type=int, default=0.8, help="Ratio of training data")
    parser.add_argument("--max_len", type=int, default=200, help="Maximum sequence length")
    parser.add_argument("--input_dim", type=int, default=4096, help="Input dimension")
    parser.add_argument("--model_dim", type=int, default=1024, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--dim_feedforward", type=int, default=4096, help="Dimension of feedforward network")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--scheduler_warmup_epochs", type=int, default=10, help="Number of epochs for linear warmup")
    parser.add_argument("--scheduler_max_epochs", type=int, default=100, help="Number of epochs for cosine annealing")
    parser.add_argument("--root_dir", type=str, default='/mnt/efs/fs1/tung_results/', help="Exp root directory")
    parser.add_argument("--resume", action="store_true", help="Path to checkpoint to resume from")
    parser.add_argument("--wandb_project", type=str, default="biohackathon", help="WandB project name")

    args = parser.parse_args()

    main(args)