import pandas as pd
import numpy as np
from Bio import SeqIO
from collections import Counter

phage_fasta = '/gladstone/engelhardt/lab/hvu/hackathonBio/phageDB/Actinobacteriophages-All.fasta'
meta_data_fn = '/gladstone/engelhardt/lab/hvu/hackathonBio/phageDB/PhagesDB_Data.txt'
phage_header_fn = '/gladstone/engelhardt/lab/hvu/hackathonBio/phageDB/phages.headers'
output_folder = '/gladstone/engelhardt/lab/hvu/hackathonBio/phageDB/phages/fasta'

meta_df = pd.read_csv(meta_data_fn, sep='\t', header = 0, index_col = None)
meta_df.rename(columns = {'Phage Name':'phageName'}, inplace = True)
phage_headers = pd.read_csv(phage_header_fn, sep='\t', header = 0, index_col = None)

def process_seqName(seqName):
    seqName = seqName.replace('Phage', 'phage')
    try:
        seqName = seqName.split('phage')[1].split()[0]
    except:
        print (f'{seq_name} cannot be resolved')
    return seqName

written_headers = Counter([])

# Parse the input FASTA file
for seq_name, record in enumerate(SeqIO.parse(phage_fasta, "fasta")):
    # Create a unique filename for each sequence record
    cleaned_seq_name = process_seqName(record.description)
    output_file = f"{output_folder}/phageDB{cleaned_seq_name}.fasta"
    # Write each sequence to a separate FASTA file
    if cleaned_seq_name in meta_df['phageName'].values:
        if cleaned_seq_name in written_headers:
            print(f"Skipped {cleaned_seq_name} in metadata")
            continue
        output_handle = open(output_file, "w")
        SeqIO.write(record, output_handle, "fasta")
        output_handle.close()
        written_headers[cleaned_seq_name] += 1
        print(f"Saved {record.id} to {output_file}")

'''
There are 3 phages for which we found more 2 different sequences in the provided FASTA file.
      2 Found Commonplace in headers
      2 Found Nonagon in headers
      2 Found Sunfish in headers
So the code will only accept the first occurence of each of these sequences
'''