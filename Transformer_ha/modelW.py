from typing import Optional

import torch
import torch.nn as nn
import math
from torch import Tensor

class MultiheadAttentionWithWeights(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True,
                 add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None, batch_first=False, **kwargs):
        super().__init__(embed_dim, num_heads, dropout=dropout, bias=bias,
                         add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn,
                         kdim=kdim, vdim=vdim, batch_first=batch_first)
        self.attn_weights = None

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None):
        output, attention = super().forward(query, key, value,
                                            key_padding_mask=key_padding_mask,
                                            need_weights=need_weights,
                                            attn_mask=attn_mask)
        self.attn_weights = attention
        if need_weights:
            return output, attention
        else:
            return output

class TransformerEncoderLayerWithWeights(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=True, **kwargs):
        super().__init__(d_model=d_model, nhead=nhead,
                         dim_feedforward=dim_feedforward, dropout=dropout,
                         activation=activation, batch_first=batch_first)
        # Replace the self-attention module with our custom one
        self.self_attn = MultiheadAttentionWithWeights(
            d_model, nhead, dropout=dropout,
            batch_first=batch_first, **kwargs
        )

    def forward(self, src, src_mask=None, src_key_padding_mask=None, need_weights=False):
        '''
        Reference: https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer
        forward function, with norm_first = False
        '''
        result = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=need_weights
        )
        if need_weights:
            src2, attn_weights = result
        else:
            src2 = result
            attn_weights = None

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src = self.norm2(src + self._ff_block(src))

        if need_weights:
            return src, attn_weights
        else:
            return src, None

    def _ff_block(self, x: Tensor) -> Tensor:
        '''
        Feed forward block, as in the original TransformerEncoderLayer
        '''
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


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

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim)
        )

        self.host_pos_encoder = PositionalEncoding(model_dim, max_len)
        self.phage_pos_encoder = PositionalEncoding(model_dim, max_len)


        self.transformer_encoder = nn.ModuleList([TransformerEncoderLayerWithWeights(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        ) for _ in range(num_layers)])

        self.norm = nn.LayerNorm(model_dim)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, 1)
        )

        # Store attention weights
        self.attention_weights = []


    def save_attention_weights(self, module, input, output):
        # output is (attn_output, attn_output_weights)
        attn_output_weights = output[1]  # Shape: [batch_size * num_heads, seq_len, seq_len]
        self.attention_weights.append(attn_output_weights)

    def get_attention_weights(self):
        """Returns attention weights from all transformer layers."""
        attention_weights = []
        for layer in self.transformer_encoder.layers:
            if hasattr(layer.self_attn, 'attn_weights'):
                attention_weights.append(layer.self_attn.attn_weights)
        return attention_weights

    def forward(self, sequences, host_len, phage_len, need_weights=False):
        self.attention_weights = []  # Reset attention weights

        # Create mask for padding
        L = sequences.size(1)
        padding_mask = self.create_padding_mask(L, host_len, phage_len, sequences.device)
        # Embed input sequences
        x = self.embedding(sequences)
        # Add separate positional encodings for host and phage
        x = self.add_separate_positional_encodings(x, host_len, phage_len)
        # Apply transformer encoder
        for layer in self.transformer_encoder:
            x, layer_attn = layer(x, src_key_padding_mask=padding_mask, need_weights=need_weights)
            self.attention_weights.append(layer_attn)
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
# max_len = 200
# input_dim = 4096
# model_dim = 1024
# num_heads = 8
# num_layers = 4
# batch_size = 1
# dim_feedforward = 4096
# dropout = 0.0
# device = 'cuda'
#
# model = TransformerClassifier(input_dim, model_dim, num_heads, num_layers, dim_feedforward)
#
# import pandas as pd
# import random
# from dataset import PairDataset
# from torch.utils.data import DataLoader
#
# phage_embed = '/gladstone/engelhardt/lab/hvu/hackathonBio/Ecoli/phages/embed'
# host_embed = '/gladstone/engelhardt/lab/hvu/hackathonBio/Ecoli/host/embed/chunk_40K'
# test_prompt_fn = '/gladstone/engelhardt/lab/hvu/hackathonBio/metadata/ecoli/train_positive_pairs.txt'
# norm_constants_fn = '/gladstone/engelhardt/lab/hvu/hackathonBio/hackathon_results/tung/tung_results/ecoli/norm_constants.pth'
# pairs_df = pd.read_csv(test_prompt_fn, sep='\t')
# pairs = [(row['Host_ID'], row['Phage_ID']) for _, row in pairs_df.iterrows()]
# dummy_labels = [-1] * len(pairs)
#
# test_dataset = PairDataset(
#     host_root=host_embed,
#     phage_root=phage_embed,
#     pairs=pairs,
#     labels=dummy_labels,
#     max_len=max_len,
#     remove_non_exist=False,
# )
#
# norm_constants = torch.load(norm_constants_fn)
# test_dataset.set_norm_constants(*norm_constants)
# dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
# for batch in dataloader:
#     sequences, labels, host_len, phage_len = batch
#     output = model(sequences, host_len, phage_len, need_weights=True)
#     print(model.attention_weights)
#     # print(output)
#     break