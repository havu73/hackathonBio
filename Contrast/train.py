import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import os
import argparse
import model

def get_train_val_dataloader(train_positive_pairs, normalization_method, batch_size, embedding_dir):
    for i in train_positive_pairs.index:
        pair = train_positive_pairs.loc[i]
        if (not os.path.isfile(embedding_dir + '/phage_embeddings/' + pair['Phage_ID'] + '.pt')) or (not os.path.isfile(embedding_dir + '/host_embeddings/' + pair['Host_ID'] + '.pt')):
            train_positive_pairs = train_positive_pairs.drop(i)
    
    unique_bacteria = train_positive_pairs['Host_ID'].unique()
    unique_phage = train_positive_pairs['Phage_ID'].unique()

    positive_pair_list = list(zip(train_positive_pairs['Host_ID'].values, train_positive_pairs['Phage_ID'].values))

    negative_pairs = []
    for bact in unique_bacteria:
        for phage in unique_phage:
            if (bact, phage) not in positive_pair_list:
                negative_pairs.append((bact, phage))
    
    negative_pairs = pd.DataFrame(negative_pairs, columns=['Host_ID', 'Phage_ID'])
    positive_pairs_split = train_positive_pairs[['Host_ID', 'Phage_ID']]
    negative_pairs['Label'] = 0
    positive_pairs_split['Label'] = 1
    full_pairs = pd.concat([positive_pairs_split, negative_pairs])
    
    train, val = train_test_split(full_pairs, test_size=0.15, random_state=42)
    print(train.shape)
    print(val.shape)    

    train_dataset = model.PhageBactPairDataset(pair_df=train, method=normalization_method, embedding_dir=embedding_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = model.PhageBactPairDataset(pair_df=val, method=normalization_method, embedding_dir=embedding_dir)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader

def train(encoder, margin, learning_rate, train_loader, val_loader, device):
    encoder = encoder.to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    criterion = model.ContrastiveLoss(margin=margin)
    
    training_loss = []
    validation_loss = []
    for epoch in range(10):
        print(epoch)
        # with tqdm.tqdm(total=len(train_loader), desc='Epoch ' + str(epoch+1)) as pbar:
        batch_train_loss = 0
        count = 0
        for batch in train_loader:
            # pbar.update(1)

            batch_phage = batch[0].to(device)
            batch_bacteria = batch[1].to(device)
            batch_labels = batch[2].to(device)
            
            batch_phage = torch.reshape(batch_phage, (batch_phage.size()[0], 1, batch_phage.size()[1], batch_phage.size()[2]))
            batch_bacteria = torch.reshape(batch_bacteria, (batch_bacteria.size()[0], 1, batch_bacteria.size()[1], batch_bacteria.size()[2]))
            
            phage_batch_embed = encoder(batch_phage)
            bact_batch_embed = encoder(batch_bacteria)
            
            train_loss = criterion(phage_batch_embed, bact_batch_embed, batch_labels)
            batch_train_loss += train_loss.item()
            count += 1
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
        training_loss.append(batch_train_loss/count)
        
        with torch.no_grad():
            batch_val_loss = 0
            count = 0
            for batch in val_loader:
                val_batch_phage = batch[0].to(device)
                val_batch_bacteria = batch[1].to(device)
                val_batch_labels = batch[2].to(device)
                
                val_batch_phage = torch.reshape(val_batch_phage, (val_batch_phage.size()[0], 1, val_batch_phage.size()[1], val_batch_phage.size()[2]))
                val_batch_bacteria = torch.reshape(val_batch_bacteria, (val_batch_bacteria.size()[0], 1, val_batch_phage.size()[1], val_batch_bacteria.size()[2]))

                phage_batch_embed_val = encoder(val_batch_phage)
                bact_batch_embed_val = encoder(val_batch_bacteria)

                val_loss = criterion(phage_batch_embed_val, bact_batch_embed_val, val_batch_labels)
                batch_val_loss += val_loss.item()
                count += 1

        validation_loss.append(batch_val_loss/count)

        print(training_loss[-1])
        print(validation_loss[-1])
            
    return encoder


if __name__ == '__main__':
    print('Started')
    parser = argparse.ArgumentParser(description="Train example")
    parser.add_argument('--embedding_dir', type=str, required=False, default='/mnt/efs/fs1/data/embeddings/vibrio_embeddings/')
    parser.add_argument('--metadata_dir', type=str, required=False, default='/mnt/efs/fs1/data/metadata/vibrio/')
    parser.add_argument('--normalization_method', type=str, required=False, default='sum')
    parser.add_argument('--batch_size', type=int, required=False, default=256)
    parser.add_argument('--margin', type=float, required=False, default=1)
    parser.add_argument('--learning_rate', type=float, required=False, default=0.001)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--device', type=str, required=False, default=1)
    
    args = parser.parse_args()
    
    train_positive_pairs = pd.read_table(args.metadata_dir + 'train_positive_pairs.txt')
    
    print('Getting train and validation dataloader')
    train_loader, val_loader = get_train_val_dataloader(train_positive_pairs=train_positive_pairs, normalization_method=args.normalization_method, batch_size=args.batch_size, embedding_dir=args.embedding_dir)

    device = torch.device('cuda:' + args.device)
    
    print('Starting training')
    trained_model = train(encoder=model.CNNEmbedding(), margin=args.margin, learning_rate=args.learning_rate, train_loader=train_loader, val_loader=val_loader, device=device)
    print('Finished training')
    torch.save(trained_model.state_dict(), args.output_file)
    print('Model saved')
