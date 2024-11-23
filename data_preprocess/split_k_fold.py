import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse



# raw_fn = '/gladstone/engelhardt/lab/hvu/hackathonBio/metadata/ecoli/raw_interction_matrix.csv'
# raw_df = pd.read_csv(raw_fn, header = 0, index_col = 0, sep = ';')
# raw_df.fillna(0, inplace=True)

def plot_clustered_heatmap_with_clusters(df, n_clusters=5):
    """
    Plots a clustered heatmap of the input DataFrame and returns the cluster labels.

    Parameters:
    df (pd.DataFrame): DataFrame with rows as bacteria, columns as phages, and integer values.
    n_clusters (int): Number of clusters to form for both rows and columns.

    Returns:
    row_clusters (pd.Series): Cluster labels for rows (bacteria).
    col_clusters (pd.Series): Cluster labels for columns (phages).
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import fcluster
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy  import linkage
    df.fillna(0, inplace=True)
    # Compute the linkage matrices for rows and columns
    # Using 'euclidean' distance and 'average' linkage method
    row_linkage = linkage(pdist(df, metric='euclidean'), method='average')
    col_linkage = linkage(pdist(df.T, metric='euclidean'), method='average')
    # Create a clustermap with the precomputed linkage matrices
    g = sns.clustermap(df, row_linkage=row_linkage, col_linkage=col_linkage,
                       cmap='viridis', figsize=(8, 8))
    # Cut the dendrograms to get the desired number of clusters
    # For rows
    row_clusters = fcluster(row_linkage, n_clusters, criterion='maxclust')
    # Map the cluster labels to the row indices after clustering
    row_order = g.dendrogram_row.reordered_ind
    row_labels = pd.Series(row_clusters[row_order], index=df.index[row_order], name='RowCluster')
    # For columns
    col_clusters = fcluster(col_linkage, n_clusters, criterion='maxclust')
    # Map the cluster labels to the column indices after clustering
    col_order = g.dendrogram_col.reordered_ind
    col_labels = pd.Series(col_clusters[col_order], index=df.columns[col_order], name='ColCluster')
    # Show the plot
    plt.show()
    # Return the cluster labels
    return row_labels, col_labels

def split_k_fold(labels, K = 5):
    '''
    labels: a list of labels, ordered based on the output of the clustering from dendorgram in the heatmap
    return: a pandas Series with index as the labels and values as the cluster number
    '''
    cluster_size = len(labels) // K
    cluster_labels = []
    for i in range(K):
        cluster_labels += [i] * cluster_size
    cluster_labels += [K-1] * (len(labels) - len(cluster_labels))
    return pd.Series(index=labels, data=cluster_labels)

def get_train_data(interaction_df, train_bacteria, train_phage, save_folder=None):
    '''
    Create train_positive_pairs.txt, train_negative_pairs.txt
    '''
    interaction_df = interaction_df.loc[train_bacteria, train_phage]
    interaction_df  = interaction_df.melt(ignore_index=False).reset_index().rename(columns={'bacteria': 'Host_ID', 'variable': 'Phage_ID', 'value': 'label'})
    # filter out nan values
    print('size of interaction_df before filtering nan values: ', interaction_df.shape)
    interaction_df = interaction_df[~interaction_df['label'].isna()]
    print ('size of interaction_df after filtering nan values: ', interaction_df.shape)
    interaction_df['label'] = interaction_df['label'].astype(int).apply(lambda x: 1 if x > 0 else 0)
    train_pos_df = interaction_df[interaction_df['label'] == 1]
    train_neg_df = interaction_df[interaction_df['label'] == 0]
    if save_folder:
        train_pos_df.to_csv(f'{save_folder}/train_positive_pairs.txt', sep='\t', index=False)
        train_neg_df.to_csv(f'{save_folder}/train_negative_pairs.txt', sep='\t', index=False)
    return train_pos_df, train_neg_df

def get_test_data(interaction_df, test_bacteria, test_phage, save_folder=None):
    '''
    Create test_prompt.txt
    '''
    interaction_df = interaction_df.loc[test_bacteria, test_phage]
    interaction_df  = interaction_df.melt(ignore_index=False).reset_index().rename(columns={'bacteria': 'Host_ID', 'variable': 'Phage_ID', 'value': 'label'})
    # filter out nan values
    print('size of interaction_df before filtering nan values: ', interaction_df.shape)
    interaction_df = interaction_df[~interaction_df['label'].isna()]
    print ('size of interaction_df after filtering nan values: ', interaction_df.shape)
    interaction_df['label'] = interaction_df['label'].astype(int).apply(lambda x: 1 if x > 0 else 0)
    if save_folder:
        interaction_df.to_csv(f'{save_folder}/test_prompt.txt', sep='\t', index=False)
    return interaction_df

def train_test_split_grid(interaction_df, bac_labels, phage_labels, K=5, output_folder=None):
    '''
    Given the interaction_df, bac_labels, phage_labels, K, output_folder:
    - Split the phages and bacterias into K clusters
    - For each combination of (phage cluster, bacteria cluster), we will keep one group for testing, the rest for training
    - For each combination of (phage cluster, bacteria cluster), put into the folder: output_folder/round_{idx}/
        - train_positive_pairs.txt: host_id, phage_id, label
        - train_negative_pairs.txt: host_id, phage_id, label
        - test_prompt.txt: host_id, phage_id, label
    '''
    from itertools import product
    import os
    os.makedirs(output_folder, exist_ok=True)
    # Split the phages and bacterias into K clusters
    bac_clusters = split_k_fold(bac_labels, K)
    phage_clusters = split_k_fold(phage_labels, K)
    for idx, (bac_k, phage_k) in enumerate(product(range(K), range(K))):
        test_bacteria = bac_clusters[bac_clusters==bac_k].index
        test_phage = phage_clusters[phage_clusters==phage_k].index
        train_bacteria = bac_clusters[bac_clusters!=bac_k].index
        train_phage = phage_clusters[phage_clusters!=phage_k].index
        round_folder = f'{output_folder}/round_{idx}'
        os.makedirs(round_folder, exist_ok=True)
        get_train_data(interaction_df, train_bacteria, train_phage, save_folder=round_folder)
        get_test_data(interaction_df, test_bacteria, test_phage, save_folder=round_folder)
        print(f'Round {idx} done')
    return

def train_test_split_columns(interaction_df, bac_labels, phage_labels, K=5, output_folder=None):
    '''
    Given the interaction_df, bac_labels, phage_labels, K, output_folder:
    - Split the phages and bacteria into K clusters
    - For each phage cluster, we will keep one group for testing, the rest for training
    - For each bacterial cluster, we will keep one group for testing, the rest for training
    - In tottal, there are K*2 rounds
    - For each round, put into the folder: output_folder/round_{idx}/
        - train_positive_pairs.txt: host_id, phage_id, label
        - train_negative_pairs.txt: host_id, phage_id, label
        - test_prompt.txt: host_id, phage_id, label
    '''
    import os
    os.makedirs(output_folder, exist_ok=True)
    # Split the phages and bacterias into K clusters
    # since we did not cluster bac_labels and phag_labels, calling to split_k_fold will be a random split
    bac_clusters = split_k_fold(bac_labels, K)
    phage_clusters = split_k_fold(phage_labels, K)
    for idx, bac_k in enumerate(range(K)):
        test_bacteria = bac_clusters[bac_clusters==bac_k].index
        train_bacteria = bac_clusters[bac_clusters!=bac_k].index
        round_folder = f'{output_folder}/round_{idx}'
        os.makedirs(round_folder, exist_ok=True)
        get_train_data(interaction_df, train_bacteria, phage_labels, save_folder=round_folder)
        get_test_data(interaction_df, test_bacteria, phage_labels, save_folder=round_folder)
        print(f'Round {idx} done')
    for idx, phage_k in enumerate(range(K)):
        test_phage = phage_clusters[phage_clusters==phage_k].index
        train_phage = phage_clusters[phage_clusters!=phage_k].index
        round_folder = f'{output_folder}/round_{K+idx}'
        os.makedirs(round_folder, exist_ok=True)
        get_train_data(interaction_df, bac_labels, train_phage, save_folder=round_folder)
        get_test_data(interaction_df, bac_labels, test_phage, save_folder=round_folder)
        print(f'Round {K+idx} done')
    return

def filter_raw_df(raw_df, phage_embed_folder, host_embed_folder):
    '''
    Filter the raw_df to only keep the rows and columns that are in bac_labels and phage_labels
    '''
    import glob
    phage_list = glob.glob(f'{phage_embed_folder}/*.pt')
    host_list = glob.glob(f'{host_embed_folder}/*.pt')
    phage_list = list(map(lambda x: x.split('/')[-1].split('.pt')[0], phage_list))
    host_list = list(map(lambda x: x.split('/')[-1].split('.pt')[0], host_list))
    # find the intersection between phage_list and host_list with the columns and rows of raw_df
    common_phages = set(phage_list) & set(raw_df.columns)
    # the phages in phage_list but not in raw_df.columns
    # redudant_phages = set(phage_list) - set(raw_df.columns)
    print('len(host_list): ', len(host_list))
    print('len(raw_df.index): ', len(raw_df.index))
    common_hosts = set(host_list) & set(raw_df.index)
    # redudant_hosts = set(host_list) - set(raw_df.index)
    # filter in raw_df only columns that are present in phage_list and rows that are present in host_list
    raw_df = raw_df.loc[list(common_hosts), list(common_phages)]
    return raw_df


def parse_arg():
    parser = argparse.ArgumentParser(description='plot clustered heatmap')
    parser.add_argument('--interaction_fn', type=str, help='data of the interaction patterns between phages and bacteria')
    parser.add_argument('--K', type=int, help='number of clusters of phages/bacteria that we will keep one group for testing')
    parser.add_argument('--phage_embed', type=str, help='path to the phage embeddings folder')
    parser.add_argument('--host_embed', type=str, help='path to the host embeddings folder')
    parser.add_argument('--output_folder', type=str, help='output file name')
    parser.add_argument('--cluster', action='store_true', help='whether to cluster the heatmap')
    # split method is either grid or column
    parser.add_argument('--split_method', type=str, default='grid', choices=['grid', 'column'], help='grid or columns')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arg()
    raw_df = pd.read_csv(args.interaction_fn, header = 0, index_col = 0, sep = '\t')
    raw_df = filter_raw_df(raw_df, args.phage_embed, args.host_embed)
    if args.cluster:
        bac_labels, phage_labels = plot_clustered_heatmap_with_clusters(raw_df.copy())
        # Plot the clustered heatmap
        bac_labels, phage_labels = plot_clustered_heatmap_with_clusters(raw_df.copy())
        print('Done clustering heatmap')
        bac_labels = bac_labels.index
        print(bac_labels)
        phage_labels = phage_labels.index
        print(phage_labels)
    else:
        bac_labels = raw_df.index
        phage_labels = raw_df.columns
    print('raw_df.shape', raw_df.shape)
    print('Total number of bacteria:', len(bac_labels))
    print('Total number of phages:', len(phage_labels))
    # Split the data into training and testing sets
    if args.split_method == 'grid':
        train_test_split_grid(raw_df, bac_labels, phage_labels, K=args.K, output_folder=args.output_folder)
    else:
        train_test_split_columns(raw_df, bac_labels, phage_labels, K=args.K, output_folder=args.output_folder)
    print('Done splitting the data')
