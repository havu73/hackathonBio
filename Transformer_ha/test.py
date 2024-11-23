import os
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from dataset import PairDataset
from model import TransformerClassifier
import helper


def main(args):
    pairs_df = pd.read_csv(args.test_prompt_fn, sep='\t')
    pairs = [(row['Host_ID'], row['Phage_ID']) for _, row in pairs_df.iterrows()]
    dummy_labels = [-1] * len(pairs)
    
    test_dataset = PairDataset(
        host_root=args.host_embed,
        phage_root=args.phage_embed,
        pairs=pairs,
        labels=dummy_labels,
        max_len=args.max_len,
    )
    norm_constants = torch.load(args.norm_constants_fn)
    test_dataset.set_norm_constants(*norm_constants)
    test_dataset.normalize_data()
    
    # load model from checkpoints/best_checkpoint.pth
    model = TransformerClassifier(
        input_dim=args.input_dim,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_len=args.max_len,
    ).cuda()

    checkpoint = torch.load(args.model_fn)

    msg = model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    print(msg)
    
    all_predictions = []
    for i in tqdm(range(len(test_dataset))):
        try:
            embeddings, _, host_lens, phage_lens = test_dataset[i]
        except:
            print (f'Could not get the test data point {i}')
            all_predictions.append(np.nan)
            continue
        if embeddings is None:
            all_predictions.append(np.nan)
            continue
        embeddings = embeddings.unsqueeze(0).cuda()
        host_lens = torch.tensor([host_lens], dtype=torch.long).cuda()
        phage_lens = torch.tensor([phage_lens], dtype=torch.long).cuda()
        outputs = model(embeddings, host_lens, phage_lens)
        all_predictions.append(outputs.item())

    print ('Done with predcitons')
    # add predictions to pairs_df as 'prediction' column
    pairs_df['prediction'] = all_predictions
    helper.create_folder_for_file(args.save_fn)
    pairs_df.to_csv(args.save_fn, sep='\t', index=False)
    print(f"Predictions saved to {args.save_fn}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer Classifier")
    parser.add_argument('--phage_embed', type=str, default='phage_embeddings', help="path to phage embeddings folder")
    parser.add_argument('--host_embed', type=str, default='host_embeddings', help="path to host embeddings folder")
    parser.add_argument('--test_prompt_fn', type=str, default='train_positive_pairs.txt',
                        help="path to test pairs")
    parser.add_argument('--save_fn', type=str, default=None, help="path to where we save the predictions")
    parser.add_argument("--model_fn", type=str, required=True, help="Path to trained model")
    parser.add_argument("--norm_constants_fn", type=str, required=True, help="Path to norm constants")
    parser.add_argument("--max_len", type=int, default=200, help="Maximum sequence length")
    parser.add_argument("--input_dim", type=int, default=4096, help="Input dimension")
    parser.add_argument("--model_dim", type=int, default=1024, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--dim_feedforward", type=int, default=4096, help="Dimension of feedforward network")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")

    args = parser.parse_args()

    main(args)