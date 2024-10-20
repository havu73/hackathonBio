import os
import pandas as pd

# Path to the meta.txt file and host_fasta folder
folder = '/home/ec2-user/phageScope/'
host_fasta_folder = f'{folder}/host_fasta'
pair_fn = f'{folder}/PhageScope_species_host_pairs.txt'
meta_file = f'{folder}/positive_pairs.txt'

# Read the host_ID column from the meta.txt file
meta_df = pd.read_csv(meta_file, sep='\t')  # Adjust the delimiter if needed
host_ids = set(meta_df['Host_ID'].astype(str))  # Convert to a set for faster lookup
# print(len(host_ids))
# print(host_ids)
# Iterate over each file in the host_fasta folder
for filename in os.listdir(host_fasta_folder):
    # print(filename)
    host_id = os.path.splitext(filename)[0]
    # print(host_id.split('.fasta'))
    host_id = host_id.split('.fasta')[0]  # Get the file name without extension
    # print (host_id)
    # If the file name does not match any host_ID, remove it
    if host_id not in host_ids:
        file_path = os.path.join(host_fasta_folder, filename)
        os.remove(file_path)
        print(f'Removed {file_path}')
