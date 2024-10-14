import os
import gzip
import glob
import torch
from tqdm import tqdm
from Bio import SeqIO
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModelForCausalLM
from evo import Evo
import argparse


def chunk_string(sequence, M):
    # Use list comprehension to chunk the string
    return [sequence[i:i+M] for i in range(0, len(sequence), M)]


class DNADataset(Dataset):
    def __init__(self, file_paths, chunk_size=40000):
        self.file_paths = file_paths
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with gzip.open(self.file_paths[idx], 'rt') as handle:
            record = list(SeqIO.parse(handle, 'fasta'))[0]
        sequence = str(record.seq)
        sequences = chunk_string(sequence, self.chunk_size)

        return sequences, os.path.basename(self.file_paths[idx])


def main(device_id, root_dir, save_dir, num_parts, part_id, chunk_size):
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    
    # do this only to use the tokenizer, we'll use HuggingFace's model for the embeddings
    evo_model = Evo('evo-1-131k-base')
    tokenizer = evo_model.tokenizer
    del evo_model

    config = AutoConfig.from_pretrained('togethercomputer/evo-1-131k-base', trust_remote_code=True, revision="1.1_fix")
    model = AutoModelForCausalLM.from_pretrained(
        'togethercomputer/evo-1-131k-base',
        config=config,
        trust_remote_code=True,
        revision="1.1_fix",
    ).to(device)
    model.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_paths = glob.glob(os.path.join(root_dir, '*.fasta.gz'))
    file_paths.sort()  # Sort to ensure reproducibility

    len_each_part = len(file_paths) // num_parts
    start_idx = part_id * len_each_part
    end_idx = (part_id + 1) * len_each_part
    file_paths = file_paths[start_idx:end_idx]

    dataset = DNADataset(file_paths, chunk_size=chunk_size)
    num_sequences = len(dataset)

    for seq_idx in tqdm(range(num_sequences), total=num_sequences, position=0, unit='sequence'):
        sequences, base_name = dataset[seq_idx]
        save_name = base_name.replace('.fasta.gz', '.pt')
        save_path = os.path.join(save_dir, save_name)

        # to avoid recomputing embeddings
        if os.path.exists(save_path):
            continue

        all_embeddings = []
        for i in tqdm(range(len(sequences)), position=1, unit='subsequence'):
            input_ids = torch.tensor(
                tokenizer.tokenize(sequences[i]), dtype=torch.long,
            ).to(device).unsqueeze(0)
            with torch.no_grad():
                x = model.backbone.embedding_layer.embed(input_ids)
                x, _ = model.backbone.stateless_forward(x)
                x = model.backbone.norm(x)
            x = x.mean(dim=1)  # (1, 4096)
            all_embeddings.append(x.float().cpu())
        all_embeddings = torch.cat(all_embeddings, dim=0)  # (L/40000, 4096)
        
        # Save embeddings
        torch.save(all_embeddings, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DNA Embedding Script")
    parser.add_argument('--device_id', type=int, default=0, help="CUDA device ID")
    parser.add_argument('--root_dir', type=str, required=True, help="Directory with input FASTA files")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory to save embeddings")
    parser.add_argument('--num_parts', type=int, default=1, help="Total number of parts to split the files")
    parser.add_argument('--part_id', type=int, default=0, help="ID of the current part being processed")
    parser.add_argument('--chunk_size', type=int, default=40000, help="Chunk size")

    args = parser.parse_args()

    main(args.device_id, args.root_dir, args.save_dir, args.num_parts, args.part_id, args.chunk_size)