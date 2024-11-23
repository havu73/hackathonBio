import torch
import torch.nn as nn
import math

class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        model_dim,
        num_heads,
        num_layers,
        dim_feedforward,
        dropout=0.1,
        max_len=1000
    ):
        super(TransformerClassifier, self).__init__()
        
        # self.embedding = nn.Linear(input_dim, model_dim)
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim)
        )
        
        self.host_pos_encoder = PositionalEncoding(model_dim, max_len)
        self.phage_pos_encoder = PositionalEncoding(model_dim, max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        self.norm = nn.LayerNorm(model_dim)
        
        # self.classifier = nn.Linear(model_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, 1)
        )

        # initialize weights
        self.attention_weights = []


        
    def forward(self, sequences, host_len, phage_len):
        # sequences: BxLxD, host_len: B, phage_len: B
        L = sequences.size(1)
        
        # Create mask for padding
        padding_mask = self.create_padding_mask(L, host_len, phage_len, sequences.device)
        
        # Embed input sequences
        x = self.embedding(sequences)
        
        # Add separate positional encodings for host and phage
        x = self.add_separate_positional_encodings(x, host_len, phage_len)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # Global average pooling over non-padding positions
        x = self.global_average_pooling(x, padding_mask)
        
        x = self.norm(x)
        
        # Classification
        output = torch.sigmoid(self.classifier(x))
        
        return output
    
    def create_padding_mask(self, L, host_len, phage_len, device):
        mask = torch.arange(L, device=device).expand(len(host_len), L) >= (host_len + phage_len).unsqueeze(1)
        return mask
    
    def add_separate_positional_encodings(self, x, host_len, phage_len):
        B, L, _ = x.shape
        device = x.device
        
        # Create position indices
        position_ids = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)
        
        # Create host/phage separation indices
        sep_indices = host_len.unsqueeze(1).expand(-1, L)
        
        # Create mask for host (True) and phage (False) positions
        host_mask = position_ids < sep_indices
        phage_mask = (position_ids >= sep_indices) & (position_ids < (host_len + phage_len).unsqueeze(1))
        
        # Get separate positional encodings for host and phage
        host_pos_enc = self.host_pos_encoder(position_ids) * host_mask.unsqueeze(-1).float()
        phage_pos_enc = self.phage_pos_encoder(position_ids - sep_indices) * phage_mask.unsqueeze(-1).float()
        
        # Combine the encodings
        pos_enc = host_pos_enc + phage_pos_enc
        
        return x + pos_enc
    
    def global_average_pooling(self, x, padding_mask):
        # Create a mask for non-padding positions
        non_pad_mask = ~padding_mask
        
        # Sum over non-padding positions
        x_sum = (x * non_pad_mask.unsqueeze(-1)).sum(dim=1)
        
        # Count non-padding positions
        non_pad_lengths = non_pad_mask.sum(dim=1).unsqueeze(-1)
        
        # Average over non-padding positions
        x_avg = x_sum / non_pad_lengths
        
        return x_avg

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        
        # Create a learnable parameter instead of a fixed buffer
        self.pe = nn.Parameter(torch.zeros(max_len, d_model), requires_grad=True)
        
        # Initialize with the original sinusoidal pattern
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.pe.data[:, 0::2] = torch.sin(position * div_term)
        self.pe.data[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        return self.pe[x]

# # Example usage
# input_dim = 4096
# model_dim = 1024
# num_heads = 8
# num_layers = 4
# dim_feedforward = 4096
# batch_size = 1

# model = TransformerClassifier(input_dim, model_dim, num_heads, num_layers, dim_feedforward)

# import pandas as pd
# import random
# from dataset import PairDataset
# from torch.utils.data import DataLoader

# positive_pairs_df = pd.read_csv('/mnt/efs/fs1/data/metadata/klebsiella/train_positive_pairs.txt', sep='\t')
# positive_pairs = [(row['Host_ID'], row['Phage_ID']) for _, row in positive_pairs_df.iterrows()]
# negative_pairs_ratio = 10
# host_ids = positive_pairs_df['Host_ID'].unique()
# phage_ids = positive_pairs_df['Phage_ID'].unique()
# negative_pairs = [
#     (host_id, phage_id) \
#         for host_id in host_ids \
#             for phage_id in phage_ids if (host_id, phage_id) not in positive_pairs
# ]

# random.shuffle(negative_pairs)
# negative_pairs = negative_pairs[:len(positive_pairs) * negative_pairs_ratio]

# all_pairs = positive_pairs + negative_pairs
# all_labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)

# dataset = PairDataset(
#     host_root='/mnt/efs/fs1/data/embeddings/klebsiella_embeddings/host_embeddings',
#     phage_root='/mnt/efs/fs1/data/embeddings/klebsiella_embeddings/phage_embeddings',
#     pairs=all_pairs,
#     labels=all_labels,
#     max_len=200
# )
# norm_constants = dataset.get_norm_constants()
# dataset.set_norm_constants(*norm_constants)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# for batch in dataloader:
#     sequences, labels, host_len, phage_len = batch
#     output = model(sequences, host_len, phage_len)
#     # print(output)
#     break