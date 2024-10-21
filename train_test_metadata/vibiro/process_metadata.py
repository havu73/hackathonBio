import pandas as pd
import numpy as np
import os
folder = '/home/ec2-user/vibrio_embeddings/'
host_folder = f'{folder}/host_embeddings/'
avail_hosts = os.listdir(host_folder)
avail_hosts = list(map(lambda x: x.split('.pt')[0], avail_hosts))
pair_fn = f'{folder}/pairs_positive.txt'
pair_df = pd.read_csv(pair_fn, sep='\t', header=0, index_col=None)
pair_df.rename(columns = {'phage_id':'Phage_ID', 'bacterial_id':'Host_ID'}, inplace = True)
# filter for only rows with host_id in avail_hosts
pair_df = pair_df[pair_df['Host_ID'].isin(avail_hosts)]  # there are, really only
miss_df = pair_df[~(pair_df['Host_ID'].isin(avail_hosts))]
#all phages are unique, so we will divide the data into train and test based on the hosts
np.random.seed(0)

# Split the data into train and test
# get the number of phages and host
phages = pair_df['Phage_ID'].unique()
hosts = pair_df['Host_ID'].unique()
print(f'phages: {len(phages)}')
print(f'hosts: {len(hosts)}')
# randomly select 20% of the phages and hosts. Then, any positive pairs between these phages and hosts will be used for testing
phages_test = np.random.choice(phages, int(0.2*len(phages)), replace=False)
hosts_test = np.random.choice(hosts, int(0.2*len(hosts)), replace=False)
print(f'phages_test: {len(phages_test)}')
print(f'hosts_test: {len(hosts_test)}')
test_pair_idx = pair_df[(pair_df['Phage_ID'].isin(phages_test)) & (pair_df['Host_ID'].isin(hosts_test))].index
pair_df['train_or_test'] = 'train'
pair_df.loc[test_pair_idx, 'train_or_test'] = 'test'
# save into separate files the positive pairs for training and testing
train_pair_df = pair_df[pair_df['train_or_test'] == 'train'][['Phage_ID', 'Host_ID', 'train_or_test']]
test_pair_df = pair_df[pair_df['train_or_test'] == 'test'][['Phage_ID', 'Host_ID', 'train_or_test']]

train_pair_df.to_csv(f'{folder}/train_positive_pairs.txt', sep='\t', index=False)
test_pair_df.to_csv(f'{folder}/test_positive_pairs.txt', sep='\t', index=False)

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