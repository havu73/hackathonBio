# DNA Sequence Embedding Generation

This script processes DNA sequences from `.fasta.gz` files, generates embeddings using the Evo transformer model, and saves them as `.pt` files.

## Arguments

- `--device_id`: The ID of the GPU device (e.g., `0`, `1`).
- `--root_dir`: Directory containing input `.fasta.gz` files.
- `--save_dir`: Directory to save the embeddings.
- `--num_parts`: Number of parts to split the data for distributed processing.
- `--part_id`: Part ID (starting from `0`) to specify which subset of data to process.

## Examples

### Single GPU

```bash
python script_name.py --device_id 0 --root_dir /path/to/fasta/files --save_dir /path/to/save/embeddings --num_parts 1 --part_id 0
```

### Multiple GPUs

For 2 GPUs:

```bash
# GPU 0
python script_name.py --device_id 0 --root_dir /path/to/fasta/files --save_dir /path/to/save/embeddings --num_parts 2 --part_id 0

# GPU 1
python script_name.py --device_id 1 --root_dir /path/to/fasta/files --save_dir /path/to/save/embeddings --num_parts 2 --part_id 1
```

You can adjust `--num_parts` and `--part_id` based on the number of GPUs.

## Output

The script generates `.pt` files containing embeddings for each `.fasta.gz` file in the `save_dir`.
