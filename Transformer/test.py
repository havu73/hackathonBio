import os
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from dataset import PairDataset
from model import TransformerClassifier


def main(args):
    assert args.dataset in ["klebsiella", "vibrio", "ecoli", "vikecoli", "phageDB", "phageScope"]
    
    if args.dataset == "klebsiella":
        args.pairs_file = "/mnt/efs/fs1/data/metadata/klebsiella/test_prompt.txt"
        args.host_embeddings_path = "/mnt/efs/fs1/data/embeddings/klebsiella_embeddings/host_embeddings"
        args.phage_embeddings_path = "/mnt/efs/fs1/data/embeddings/klebsiella_embeddings/phage_embeddings"
    elif args.dataset == "vibrio":
        args.pairs_file = "/mnt/efs/fs1/data/metadata/vibrio/test_prompt.txt"
        args.host_embeddings_path = "/mnt/efs/fs1/data/embeddings/vibrio_embeddings/host_embeddings"
        args.phage_embeddings_path = "/mnt/efs/fs1/data/embeddings/vibrio_embeddings/phage_embeddings"
    elif args.dataset == "ecoli":
        args.pairs_file = "/mnt/efs/fs1/data/metadata/ecoli/test_prompt.txt"
        args.host_embeddings_path = "/mnt/efs/fs1/data/embeddings/ecoli_embeddings/host_embeddings"
        args.phage_embeddings_path = "/mnt/efs/fs1/data/embeddings/ecoli_embeddings/phage_embeddings"
    elif args.dataset == "vikecoli":
        args.pairs_file = "/mnt/efs/fs1/data/metadata/vikecoli/test_prompt.txt"
        args.host_embeddings_path = "/mnt/efs/fs1/data/embeddings/vikecoli_embeddings/host_embeddings"
        args.phage_embeddings_path = "/mnt/efs/fs1/data/embeddings/vikecoli_embeddings/phage_embeddings"
    elif args.dataset == "phageDB":
        args.pairs_file = "/mnt/efs/fs1/data/metadata/phageDB/test_prompt.txt"
        args.host_embeddings_path = "/mnt/efs/fs1/data/embeddings/phageDB_embeddings/host_embeddings"
        args.phage_embeddings_path = "/mnt/efs/fs1/data/embeddings/phageDB_embeddings/phage_embeddings"
    elif args.dataset == "phageScope":
        args.pairs_file = "/mnt/efs/fs1/data/metadata/phageScope/test_prompt.txt"
        args.host_embeddings_path = "/mnt/efs/fs1/data/embeddings/phageScope_embeddings/host_embeddings"
        args.phage_embeddings_path = "/mnt/efs/fs1/data/embeddings/phageScope_embeddings/phage_embeddings"

    pairs_df = pd.read_csv(args.pairs_file, sep='\t')
    pairs = [(row['Host_ID'], row['Phage_ID']) for _, row in pairs_df.iterrows()]
    dummy_labels = [-1] * len(pairs)
    
    test_dataset = PairDataset(
        host_root=args.host_embeddings_path,
        phage_root=args.phage_embeddings_path,
        pairs=pairs,
        labels=dummy_labels,
        max_len=args.max_len,
        remove_non_exist=False,
    )
    norm_constants = torch.load(os.path.join(args.model_dir, 'norm_constants.pth'))
    test_dataset.set_norm_constants(*norm_constants)
    
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

    if args.load_epoch == -1:
        print("Loading best F1 checkpoint")
        checkpoint = torch.load(os.path.join(args.model_dir, 'checkpoints', 'best_checkpoint.pth'))
    else:
        print(f"Loading epoch {args.load_epoch}")
        checkpoint = torch.load(os.path.join(args.model_dir, 'checkpoints', f'epoch_{args.load_epoch}.pth'))

    msg = model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    print(msg)
    
    all_predictions = []
    for i in tqdm(range(len(test_dataset))):
        embeddings, _, host_lens, phage_lens = test_dataset[i]
        if embeddings is None:
            all_predictions.append(None)
            continue
        embeddings = embeddings.unsqueeze(0).cuda()
        host_lens = torch.tensor([host_lens], dtype=torch.long).cuda()
        phage_lens = torch.tensor([phage_lens], dtype=torch.long).cuda()
        outputs = model(embeddings, host_lens, phage_lens)
        all_predictions.append(outputs.item())
    
    # add predictions to pairs_df as 'prediction' column
    pairs_df['prediction'] = all_predictions
    os.makedirs(args.root_dir, exist_ok=True)
    save_path = os.path.join(args.root_dir, f'{args.dataset}_predictions.csv')
    pairs_df.to_csv(save_path, sep='\t', index=False)
    print(f"Predictions saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer Classifier")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to trained model")
    parser.add_argument("--load_epoch", type=int, required=True, help="Epoch to load")
    parser.add_argument("--root_dir", type=str, default='/mnt/efs/fs1/tung_predictions/best_f1_model', help="Exp root directory")
    parser.add_argument("--max_len", type=int, default=200, help="Maximum sequence length")
    parser.add_argument("--input_dim", type=int, default=4096, help="Input dimension")
    parser.add_argument("--model_dim", type=int, default=1024, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--dim_feedforward", type=int, default=4096, help="Dimension of feedforward network")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")

    args = parser.parse_args()

    main(args)