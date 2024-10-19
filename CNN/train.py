import os
from os.path import join
from tqdm import tqdm
import logging
import itertools
import argparse

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

from model import BacteriaNN, ViralNN, PhINN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)
    
def pad_sequence_to_max_length(sequence, max_length):
    # sequence shape: (Length, Dim)
    length, dim = sequence.shape
    if length < max_length:
        padding_amount = max_length - length
        padded_sequence = F.pad(sequence, (0, 0, 0, padding_amount))
    else:
        padded_sequence = sequence[:max_length, :]
    return padded_sequence


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on a dataset")
    parser.add_argument("--dataset", type=str, default="phageScope",
                        help="Dataset type")
    parser.add_argument("--viral_diff", type=bool, default=False, 
                        help="Use different architecture for viral")
    parser.add_argument("--model_save_out", type=str, default="models",
                        help="Path to save the model")
    parser.add_argument("--log_out", type=str, default="logs",
                        help="Path to save the logs")
    parser.add_argument("--cuda", type=int, default=0,
                        help="Cuda device")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset == "kleb":
        host_embeddings_folder = "/mnt/efs/fs1/data/embeddings/klebsiella_embeddings/host_embeddings"
        phage_embeddings_folder = "/mnt/efs/fs1/data/embeddings/klebsiella_embeddings/phage_embeddings"
        train_positive_pairs = "/mnt/efs/fs1/data/metadata/klebsiella/train_positive_pairs.txt"
    elif args.dataset == "vibrio":
        host_embeddings_folder = "/mnt/efs/fs1/data/embeddings/vibrio_embeddings/host_embeddings"
        phage_embeddings_folder = "/mnt/efs/fs1/data/embeddings/vibrio_embeddings/phage_embeddings"
        train_positive_pairs = "/mnt/efs/fs1/data/metadata/vibrio/train_positive_pairs.txt"
    elif args.dataset == "phageDB":
        host_embeddings_folder = "/mnt/efs/fs1/data/embeddings/phageDB_embeddings/host_embeddings"
        phage_embeddings_folder = "/mnt/efs/fs1/data/embeddings/phageDB_embeddings/phage_embeddings"
        train_positive_pairs = "/mnt/efs/fs1/data/metadata/phageDB/train_positive_pairs.txt"
    elif args.dataset == "phageScope":
        host_embeddings_folder = "/mnt/efs/fs1/data/embeddings/phageScope_embeddings/host_embeddings"
        phage_embeddings_folder = "/mnt/efs/fs1/data/embeddings/phageScope_embeddings/phage_embeddings"
        train_positive_pairs = "/mnt/efs/fs1/data/metadata/phageScope/train_positive_pairs.txt"
    elif args.dataset == "ecoli":        
        host_embeddings_folder = "/mnt/efs/fs1/data/embeddings/ecoli_embeddings/host_embeddings"
        phage_embeddings_folder = "/mnt/efs/fs1/data/embeddings/ecoli_embeddings/phage_embeddings"
        train_positive_pairs = "/mnt/efs/fs1/data/metadata/ecoli/train_positive_pairs.txt"
    elif args.dataset == "vikecoli":
        host_embeddings_folder = "/mnt/efs/fs1/data/embeddings/vikecoli_embeddings/host_embeddings"
        phage_embeddings_folder = "/mnt/efs/fs1/data/embeddings/vikecoli_embeddings/phage_embeddings"
        train_positive_pairs = "/mnt/efs/fs1/data/metadata/vikecoli/train_positive_pairs.txt"

    df_pos = pd.read_csv(train_positive_pairs, sep = "\t").drop(columns = ["train_or_test"])
    bacteria = df_pos["Host_ID"].unique()
    phage = df_pos["Phage_ID"].unique()
    all_possible_pairs = set(itertools.product(phage, bacteria))
    existing_pairs = set(df_pos.itertuples(index=False, name=None))
    negative_pairs = all_possible_pairs - existing_pairs
    df_neg = pd.DataFrame(list(negative_pairs), columns=df_pos.columns)
    df_neg = df_neg.sample(n=len(df_pos)*2, random_state=42)
    df_pos["label"] = 1.
    df_neg["label"] = 0.
    df_all = pd.concat([df_pos, df_neg]).drop_duplicates().sample(frac = 1., random_state=42)
    
    host_embeddings_dict = {}
    phage_embeddings_dict = {}
    for he in os.listdir(host_embeddings_folder):
        full_path = join(host_embeddings_folder, he)
        host_embeddings_dict[he.rsplit(".", 1)[0]] = torch.load(full_path)
    for pe in os.listdir(phage_embeddings_folder):
        full_path = join(phage_embeddings_folder, pe)
        pe = pe.replace("phageDB", "")
        phage_embeddings_dict[pe.rsplit(".", 1)[0]] = torch.load(full_path)
        
    viral_lengths = []
    bacterial_lengths = []
    for key, val in host_embeddings_dict.items():
        bacterial_lengths.append(val.shape[0])
    for key, val in phage_embeddings_dict.items():
        viral_lengths.append(val.shape[0])
        
    # set max length to 90th percentile
    MAX_BACTERIA_LENGTH = max(int(np.percentile(bacterial_lengths, 90)), 5)
    MAX_VIRAL_LENGTH = max(int(np.percentile(viral_lengths, 90)), 5)
    
    for key, val in host_embeddings_dict.items():
        host_embeddings_dict[key] = pad_sequence_to_max_length(val, MAX_BACTERIA_LENGTH)
    for key, val in phage_embeddings_dict.items():
        phage_embeddings_dict[key] = pad_sequence_to_max_length(val, MAX_VIRAL_LENGTH)
    
    df_all = df_all[df_all["Phage_ID"].isin(phage_embeddings_dict.keys())]
    df_all = df_all[df_all["Host_ID"].isin(host_embeddings_dict.keys())]
    
    df_train, df_val_test = train_test_split(df_all, test_size=0.3, random_state=42)
    df_val, df_test = train_test_split(df_val_test, test_size=0.5, random_state=42)
    print(df_train["label"].value_counts())
    print(df_val["label"].value_counts())
    print(df_test["label"].value_counts())
    
    embedding_dim = 4096
    embedding_reduction_factor = 2
    kernel_size = 5 # set to odd number smaller than embedding size
    padding = (kernel_size - 1) // 2
    stride = 2
    avg_poolsize = 1
    resblock_num = 1
    output_dim = 256
    dropout_rate = 0.2
    hidden_dims = [embedding_dim//2, embedding_dim//4]
    
    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"

    # Setup logging
    logging.basicConfig(level=logging.INFO, filename=args.log_out, filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Initialize models
    bacteria_nn = BacteriaNN(embedding_dim=embedding_dim, kernel_size=kernel_size,
                            padding=padding, stride=stride, avg_poolsize=avg_poolsize,
                            resblock_num=resblock_num, embedding_reduction_factor=embedding_reduction_factor,
                            output_dim=output_dim)

    if args.viral_diff:
        viral_nn = ViralNN(embedding_dim=embedding_dim, length=MAX_VIRAL_LENGTH,
                    output_dim=output_dim, hidden_dims=hidden_dims, dropout_rate=dropout_rate)
    else:
        viral_nn = BacteriaNN(embedding_dim = embedding_dim, kernel_size = kernel_size, padding = padding, stride = stride, avg_poolsize = avg_poolsize, resblock_num = resblock_num, embedding_reduction_factor = embedding_reduction_factor, output_dim = output_dim)

    phinn_model = PhINN(bacteria_nn, viral_nn, output_dim=output_dim,
                        embedding_dim=embedding_dim, skip_connection=True, skip_only=False).to(device)
    # phinn_model = nn.DataParallel(phinn_model, device_ids=[0, 1, 2])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(phinn_model.parameters(), lr=0.0001)

    num_epochs = 200
    batch_size = 64

    # Initialize metrics
    train_losses, val_losses = [], []
    best_val_loss = float('inf')  # Initialize with a large value

    for epoch in range(num_epochs):
        # Training phase
        phinn_model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0

        for i in tqdm(range(0, len(df_train), batch_size)):
            df_subset = df_train.iloc[i:i+batch_size]
            bacteria_batch = torch.stack([host_embeddings_dict[host_id] for host_id in df_subset["Host_ID"]]).to(device)
            viral_batch = torch.stack([phage_embeddings_dict[phage_id] for phage_id in df_subset["Phage_ID"]]).to(device)
            labels_batch = torch.tensor(df_subset["label"].tolist()).float().unsqueeze(dim=1).to(device)

            optimizer.zero_grad()  # Zero the gradients
            outputs = phinn_model(bacteria_batch, viral_batch)  # Forward pass
            loss = criterion(outputs, labels_batch)  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize

            train_loss += loss.item()
            predicted_train = (outputs.squeeze() > 0.5).float()
            correct_train += (predicted_train == labels_batch.squeeze()).sum().item()
            total_train += labels_batch.size(0)
            
            # delete data to save memory
            del bacteria_batch 
            del viral_batch
            del labels_batch

        avg_train_loss = train_loss / (len(df_train) // batch_size)
        train_accuracy = correct_train / total_train
        train_losses.append(avg_train_loss)

        # Validation phase
        phinn_model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0

        with torch.no_grad():
            for i in range(0, len(df_val), batch_size):
                df_subset = df_val.iloc[i:i+batch_size]
                bacteria_batch = torch.stack([host_embeddings_dict[host_id] for host_id in df_subset["Host_ID"]]).to(device)
                viral_batch = torch.stack([phage_embeddings_dict[phage_id] for phage_id in df_subset["Phage_ID"]]).to(device)
                labels_batch = torch.tensor(df_subset["label"].tolist()).float().unsqueeze(dim=1).to(device)

                outputs = phinn_model(bacteria_batch, viral_batch)
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item()

                predicted_val = (outputs.squeeze() > 0.5).float()
                correct_val += (predicted_val == labels_batch.squeeze()).sum().item()
                total_val += labels_batch.size(0)
                
                # delete data to save memory
                del bacteria_batch
                del viral_batch
                del labels_batch

        avg_val_loss = val_loss / (len(df_val) // batch_size)
        val_accuracy = correct_val / total_val
        val_losses.append(avg_val_loss)

        # Log training and validation metrics
        logging.info(f"Epoch {epoch + 1}/{num_epochs}: "
                    f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(phinn_model, args.model_save_out)
            logging.info("Best model saved with val loss: {:.4f}".format(best_val_loss))

    # Test the model
    # use best model 
    phinn_model = torch.load(args.model_save_out).to(device)
    phinn_model.eval()
    test_loss, correct_test, total_test = 0.0, 0, 0

    with torch.no_grad():
        for i in range(0, len(df_test), batch_size):
            df_sub = df_test.iloc[i:i + batch_size]
            bacteria_batch = torch.stack([host_embeddings_dict[host_id] for host_id in df_sub["Host_ID"]]).to(device)
            viral_batch = torch.stack([phage_embeddings_dict[phage_id] for phage_id in df_sub["Phage_ID"]]).to(device)
            labels_batch = torch.tensor(df_sub["label"].tolist()).float().unsqueeze(dim=1).to(device)

            outputs = phinn_model(bacteria_batch, viral_batch)
            loss = criterion(outputs, labels_batch)
            test_loss += loss.item()

            predicted_test = (outputs.squeeze() > 0.5).float()
            correct_test += (predicted_test == labels_batch.squeeze()).sum().item()
            total_test += labels_batch.size(0)
            
            # delete data to save memory
            del bacteria_batch
            del viral_batch
            del labels_batch
            

    # Calculate final test loss and accuracy
    avg_test_loss = test_loss / (len(df_test) // batch_size)
    test_accuracy = correct_test / total_test

    # Log test metrics
    logging.info(f"Final Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
if __name__ == "__main__":
    main()
    