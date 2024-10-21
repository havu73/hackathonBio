import pandas as pd
import numpy as np
import os
folder = '/home/ec2-user/phageScope/'
pair_fn = f'{folder}/PhageScope_species_host_pairs.txt'
pair_df = pd.read_csv(pair_fn, sep='\t', header=0, index_col=None)

#for each host_name, retain only one row with the highest totalUngappedLength
host_fn = f'{folder}/RefSeq_species_host_meta.txt'
host_df = pd.read_csv(host_fn, sep='\t', header=0, index_col=None)
host_df.rename(columns={'Host_ncbi_accession': 'Host_ID'}, inplace=True)
all_hosts = host_df['Host_name'].unique()
# for each host_name, retain only one row with the highest totalUngappedLength
short_host_df = host_df[host_df['Host_numberOfContigs']== 1]
short_host_df = short_host_df.loc[short_host_df.groupby('Host_name')['totalUngappedLength'].idxmax()]
short_host_df = short_host_df[['Host_name', 'Host_ID']]
# there are 140 hosts remaining (Host_numberOfContigs==1)
curr_host = short_host_df['Host_name'].values
from collections import Counter
curr_host = Counter(curr_host)
remain_host = np.setdiff1d(all_hosts, curr_host)
# for the remaining host, choose the one with the highest totalUngappedLength
print(len(all_hosts), len(curr_host), len(remain_host))

for host in all_hosts:
    if host not in curr_host:
        this_host_df = host_df[host_df['Host_name']==host]
        idx = this_host_df['totalUngappedLength'].idxmax()
        short_host_df.loc[idx] = this_host_df.loc[idx]

short_host_df = short_host_df.reset_index(drop=True)
# merge pair_df and short_host_df based on Host_name
pair_df = pair_df.merge(short_host_df, on='Host_name', how='left')
pair_df['train_or_test'] = 'train'
# group by host_id and then count the number of phages: number of phages per host range from 1 to 49x
# count_df = pair_df.groupby('Host_ID').size().sort_values(ascending=False)
# count_df = count_df.reset_index()
# count_df.columns = ['Host_ID', 'count']

# pick random host until the culmulative number of phages is between 400 and 600.
np.random.seed(0)
num_test_points = 0
while True:
    host_id = np.random.choice(short_host_df['Host_ID'].values)
    this_df = pair_df[pair_df['Host_ID']==host_id]
    if this_df.shape[0] > 400:
        continue
    if num_test_points + this_df.shape[0] < 600 and num_test_points + this_df.shape[0] > 400:
        pair_df.loc[this_df.index, 'train_or_test'] = 'test'
        num_test_points += this_df.shape[0]
        break
    if num_test_points + this_df.shape[0] > 600:
        # do not choose this host
        continue
    if num_test_points + this_df.shape[0] < 400:
        # choose this host
        pair_df.loc[this_df.index, 'train_or_test'] = 'test'
        num_test_points += this_df.shape[0]

print (f"All {pair_df.shape[0]} phages are unique. Among them:")
print(f"Train: {pair_df[pair_df['train_or_test'] == 'train'].shape[0]}")
print(f"Test: {pair_df[pair_df['train_or_test'] == 'test'].shape[0]}")

positive_pair_df = pair_df[['Phage_ID', 'Host_ID', 'train_or_test']]
train_pair_df = positive_pair_df[positive_pair_df['train_or_test'] == 'train']
test_pair_df = positive_pair_df[positive_pair_df['train_or_test'] == 'test']
positive_pair_df.to_csv(f'{folder}/positive_pairs.txt', index=False, sep = '\t')
train_pair_df.to_csv(f'{folder}/train_positive_pairs.txt', index=False, sep = '\t')
test_pair_df.to_csv(f'{folder}/test_positive_pairs.txt', index=False, sep = '\t')

def get_negative_pairs(positive_pair_df, num_negative_pairs=1, train_or_test='test'):
    positive_pair_df['label'] = 1
    hosts = positive_pair_df['Host_ID'].unique()
    phages = positive_pair_df['Phage_ID'].unique()
    neg_test_pair_df = pd.DataFrame(columns=['Phage_ID', 'Host_ID', 'train_or_test'])
    from itertools import product
    all_possible_negative_pairs = set(product(phages, hosts))
    positive_pairs = set(zip(positive_pair_df['Phage_ID'], positive_pair_df['Host_ID']))
    all_possible_negative_pairs = all_possible_negative_pairs - positive_pairs
    all_possible_negative_pairs = list(all_possible_negative_pairs)
    num_negative_pairs = min(num_negative_pairs, len(all_possible_negative_pairs))  # only sample as much as we can
    sample_neg_pairs = np.random.choice(len(all_possible_negative_pairs), num_negative_pairs, replace=False)
    for idx in sample_neg_pairs:
        phage_id, host_id = all_possible_negative_pairs[idx]
        neg_test_pair_df.loc[neg_test_pair_df.shape[0]] = {'Phage_ID': phage_id, 'Host_ID': host_id, 'train_or_test': train_or_test}
    neg_test_pair_df['label'] = 0
    return neg_test_pair_df

num_test_neg_pairs = test_pair_df.shape[0]* 10
neg_test_pair_df = get_negative_pairs(test_pair_df, num_negative_pairs=num_test_neg_pairs, train_or_test='test')
test_results_df = pd.DataFrame(columns=['Phage_ID', 'Host_ID', 'train_or_test', 'label'])
test_pair_df['label']=1
test_results_df = pd.concat([test_pair_df, neg_test_pair_df], axis=0)
test_result_df = test_results_df.reset_index(drop=True)
test_result_df.to_csv(f'{folder}/test_ground_truth.txt', index=False, sep = '\t', header = True)
test_prompt_df = test_result_df[['Phage_ID', 'Host_ID']]
test_prompt_df.to_csv(f'{folder}/test_prompt.txt', index=False, sep = '\t', header = True)