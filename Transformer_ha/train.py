import os
import random
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import wandb
from tqdm import tqdm
from dataset import PairDataset, create_dataloaders, BalancedBatchSampler
from model import TransformerClassifier
from lr_scheduler import LinearWarmupCosineAnnealingLR


random.seed(42)
torch.manual_seed(42)
np.random.seed(42)



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
            all_preds.extend(outputs.cpu().numpy() > 0.5) # prediction scores are calculated based on predictions in binary form
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy, precision, recall, f1, roc_auc = get_metrics(all_labels, all_preds)

    return avg_loss, accuracy, precision, recall, f1, roc_auc

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.root_dir = os.path.join(args.root_dir, args.exp_name)
    args.checkpoint_dir = os.path.join(args.root_dir, args.exp_name, "checkpoints")
    
    os.makedirs(args.root_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

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
        use_pos_encodings=args.use_pos_enc,
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
    if args.resume != None:
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
    parser.add_argument('--phage_embed', type=str, default='phage_embeddings', help="path to phage embeddings folder")
    parser.add_argument('--host_embed', type=str, default='host_embeddings', help="path to host embeddings folder")
    parser.add_argument('--train_positive_fn', type=str, default='train_positive_pairs.txt',
                        help="path to positive pairs")
    parser.add_argument('--train_negative_fn', type=str, default=None, help="path to negative pairs")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--negative_pairs_ratio", type=int, default=-1, help="Ratio of negative to positive pairs. If -1 then all negative pairs are sampled from during model training.")
    parser.add_argument("--train_ratio", type=int, default=0.8, help="Ratio of training data")
    parser.add_argument("--max_len", type=int, default=200, help="Maximum sequence length")
    parser.add_argument("--use_pos_enc", action="store_true", help="Use positional encoding")
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
    parser.add_argument("--resume", default= None, type = str, help="Path to checkpoint to resume from")
    parser.add_argument("--wandb_project", type=str, default="biohackathon", help="WandB project name")
    args = parser.parse_args()
    main(args)