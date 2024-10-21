import numpy as np
import pandas as pd
from Bio import SeqIO
from itertools import product
import pickle
from tqdm import tqdm
from typing import Dict, List
from collections import namedtuple
from tabulate import tabulate
from colorama import init, Fore
import os
import gzip

# gets all sequence permutations of length 4（repeats included）and makes the index of each kmer
# key order matters here
KMER_IDX_DICT = {
    ''.join(K): idx for K, idx in zip(product(['A', 'T', 'G', 'C'], repeat=4), range(256))
}
# relative path of trained models
# modify if needed
RF_PATH = './PBLKS/PBLKS_model.pkl'
XGB_PATH = './models/XGBoost_model.pkl'


class InputFileFormatError(Exception):
    '''
    This error'll be thrown if content of the input file is not orgnized in fasta format
    '''
    pass


class DirectoryError(Exception):
    '''
    This error'll be thrown if a file is provided when expecting a directory 
    '''
    pass


def _parse_sequence(sequence: str
                    ) -> np.ndarray:  # shape: [len(sequence) - kmer_with_N, 256]
    '''calculates the number of each type of 4mers(without N)
    the result is orgnized to a 2-dim np array of shape [len(sequence) - kmer_with_N, 256]
    type of 4mer starts with each nt is one-hot encoded on the second dimension
    '''
    index_list = []
    seq_length = len(sequence)
    for i in range(seq_length - 4):
        kmer = sequence[i: i + 4]
        if kmer in KMER_IDX_DICT.keys():
            index_list.append(KMER_IDX_DICT[kmer])
        else:  # if this Kmer includes N
            seq_length -= 1
    result = np.zeros([seq_length - 4, 256])
    result[np.arange(seq_length - 4), np.array(index_list)] = 1
    return result


def _open_fasta_file(fasta_path):
    # Check if the file ends with .gz to determine if it's compressed
    if fasta_path.endswith(".gz"):
        return gzip.open(fasta_path, "rt")
    else:
        return open(fasta_path, "rt")

def _countkmer(fasta_path: str,
               description: str) -> (Dict[str, List[int]]):
    '''Counts the number of kmer in each sequence in the given fasta file
    Args:
        fasta_path: path to the fasta file
        description: words used infront of progress bar
    Returns:
        kmer_dic: a dictionary, with description of sequence part as key and 
            a list including the number of each kmer as value
    '''
    kmer_dic = {}
    contig_n = 0
    # counts kmer in each sequence longer than 9000
    # scans the sequence in a window of length 9000 and step of length 1800
    # the last 9000nt of the sequence is also scanned separately
    pbar = tqdm(desc=description)
    try:
        fasta_file = _open_fasta_file(fasta_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f'{fasta_path} does not exist')
    for record in SeqIO.parse(fasta_file, "fasta"):
        pbar.update()
        seq = record.seq
        seq_length = len(seq)

        if seq_length >= 9000:
            contig_n += 1
            # the kmer type of each base and next 3 nts
            # shape: [sequence length(kmer with N is not included), 256]
            kmer_indexs = _parse_sequence(seq)
            start_indexs = np.arange(0, seq_length, 1800)

            for i in start_indexs:
                if i + 9000 <= seq_length:
                    kmer_cnt = np.sum(kmer_indexs[i: i + 9000, :], axis=0)
                    k_num = str(int(i / 1800))
                    kmer_dic[str(contig_n) + '+' + k_num] = kmer_cnt
                else:
                    break

            length = int(seq_length - 9000)
            k_num = str(int(length // 1800 + 1))
            kmer_dic[str(contig_n) + '+' +
                     k_num] = np.sum(kmer_indexs[-9000:, :], axis=0)

    # if none of the sequences in the fasta file has a length beyond 9000nt
    # count mers of length 4 in the longest sequence
    if contig_n == 0:
        max_contig = ''
        # import pdb; pdb.set_trace()
        for record in SeqIO.parse(fasta_file, "fasta"):
            # import pdb; pdb.set_trace()
            seq = record.seq
            seq_length = len(seq)
            if seq_length > len(max_contig):
                max_contig = seq
        kmer_indexs = _parse_sequence(max_contig)
        kmer_dic['1+0'] = np.sum(kmer_indexs, axis=0)
    return (kmer_dic)


def _kmerdict2feas(dic_phage: Dict[str, List[int]],
                   dic_bac: Dict[str, List[int]]) -> np.ndarray:
    '''
    find the sequence part pair in dic_phage and dic_bac that has the highest correlation
    in terms of each kemer count 
    subtract the number of each kmer in the chosen pair as model input
    Args:
        dic_phage: a dictionary, with description of sequence part as key and 
            a list including the number of each kmer as value
        dic_bac: similar as dic_phage, a dictionary, with description of sequence part 
            as key and a list including the number of each kmer as value
    Returns:
        the number of each kmer after subtraction
    '''
    p_lst = [p_key for p_key in dic_phage.keys()]
    b_lst = [b_key for b_key in dic_bac.keys()]

    my_corr = {}
    for m in range(len(p_lst)):
        for n in range(len(b_lst)):
            x = dic_phage[p_lst[m]]
            y = dic_bac[b_lst[n]]
            my_corr[p_lst[m] + '_' + b_lst[n]] = np.corrcoef(x, y)[0][-1]

    max_cor = -1
    for test_key in my_corr.keys():
        if my_corr[test_key] > max_cor:
            max_cor = my_corr[test_key]
            max_lne = test_key

    phage_lne = str(max_lne.split('_')[0])
    bac_lne = str(max_lne.split('_')[1])
    # subtract the number of each kmer in the sequence pair as model input
    pb_sub = np.array(dic_phage[phage_lne]) - np.array(dic_bac[bac_lne])
    return pb_sub.reshape(1, -1)


def get_descriptor(bac_path: str,
                   phage_path: str) -> np.ndarray:
    '''
    gets descriptors described in the paper  
    "PB-LKS: a python package for predicting Phage-Bacteria interaction through Local K-mer Strategy"
    Args:
        bac_path: path to the sequence file of Bac(content must be orgnized in fasta format)
        phage_path: path to the sequence file of Phage(content must be orgnized in fasta format)
    Returns:
        descriptors extracted from both sequence files
    Raises:
        FileNotFoundError: if at least one of the sequence file you entered does not exist
        InputFileFormatError: if the content of the sequence file is not orgnized in fasta format
    '''
    if os.path.exists(bac_path) and os.path.exists(phage_path):
        try:
            dic_bac = _countkmer(
                bac_path, description="parsing bacteria sequence")
            dic_phage = _countkmer(
                phage_path, description="parsing phage sequence")
        except Exception:
            raise InputFileFormatError('something wrong happened when parsing sequence file :('
                                       + '\n This error is very likly to be caused by the sequence file content, '
                                       + 'please make sure it is orgnized in fasta format')
        return _kmerdict2feas(dic_phage, dic_bac)
    else:
        raise FileNotFoundError(
            f'the sequence file you entered does not exist :(, {bac_path}, {phage_path}')


def _get_model(use_xgb: bool):
    '''returns the model per se
    if use_xgb is True, returns XGBoost based model; esle, returns randomforest based model
    '''
    if use_xgb:
        model_path = open(XGB_PATH, "rb")
    else:
        model_path = open(RF_PATH, "rb")
    return pickle.load(model_path)


Important_feature = namedtuple(
    "Important_feature", ["name", "value", "importance"])


def _get_important_features(model, top_cnt: int,
                            # [1, 256]
                            descriptors: np.ndarray) -> List[Important_feature]:
    '''returns {top_cnt} most important features name and their corresponding vlaue 
    from all descriptors 
    the features are in descending order of importance
    Args: 
        model: the model to be used
        top_cnt: feature number to be showed
        descriptors: raw descriptors 
    '''
    # numpy array, shape: [1, 256]
    feature_importances = model.feature_importances_
    kmer_types = [''.join(kmer)
                  for kmer in product(['A', 'T', 'G', 'C'], repeat=4)]
    result = []
    sorted_indices = np.argsort(feature_importances)[-top_cnt:]

    for index in reversed(sorted_indices.tolist()):
        result.append(Important_feature(
            kmer_types[index], descriptors[0][index], feature_importances[index]))
    return result


def show_feature_importance(top_cnt=10,
                            use_xgboost=False) -> dict[str, float]:
    '''returns the importance of {top_cnt} most important features
    Args:
        top_cnt: number of most important features
        use_xgboost: use XGBoost based model(randomforest by default)
    Returns:
        feature_importance {kmer: importance}
    '''
    model = _get_model(use_xgboost)
    # numpy array, shape: [1, 256]
    feature_importance = model.feature_importances_
    feature_importance = {
        kmer: importance for kmer, importance in zip(KMER_IDX_DICT.keys(), feature_importance)
    }
    sorted_items = sorted(feature_importance.items(),
                          key=lambda x: x[1], reverse=True)
    top_n_kmers = sorted_items[: top_cnt]
    return {kmer: importance for _, (kmer, importance) in enumerate(top_n_kmers)}


def predict(bac_path: str,
            phage_path: str,
            use_xgboost=False,
            need_detail=False,
            top_cnt=10):
    '''
    make prediction with given sequence file using given model
    for more detail about PBLKS, please check  paper:
    "PB-LKS: a python package for predicting Phage-Bacteria interaction through Local K-mer Strategy" 
    Args:
        bac_path: path to the sequence file of Bac(content must be orgnized in fasta format)
        phage_path: path to the sequence file of Phage(content must be orgnized in fasta format)
        model: the model to be used
        use_xgboost: use XGBoost based model instead of randomforest by default
        need_detail: if True, also provide descriptors of most importance
        top_number:(optional) number of most important features to show
    Returns:
        precict_result: the result(1/0) given by the model
        probability: the probability that phage and bacteria interacts given by the model
    Raises:
        FileNotFoundError: if at least one of the sequence file you entered does not exist
        InputFileFormatError: if the content of the sequence file is not orgnized in fasta format
    '''
    descriptor = get_descriptor(bac_path, phage_path)
    model = _get_model(use_xgboost)
    probability = model.predict_proba(descriptor)[0][1]
    result = int(probability > 0.5)

    if need_detail:
        important_features = _get_important_features(
            model, top_cnt, descriptor)
        return (result, probability, _orgnize_features(important_features))
    else:
        return (result, probability)


def show_decision_path(bac_path: str,
                       phage_path: str,
                       use_xgboost=False):
    '''
    shows the decision path of model
    the result is given as a list of lists, where 0,1 indicates wheter the corresponding node is passed
    when making this decision
    users can also check decision path according to visualized decision Trees after creating pictures with 
    given script
    '''
    descriptor = get_descriptor(bac_path, phage_path)
    model = _get_model(use_xgboost)

    (indicator, n_nodes_ptr) = model.decision_path(descriptor)
    print('this is the decision path of model: ')
    print('0,1 indicates whether corresponding node is passed when making decision')
    print('note that tree nodes are numbered by pre-order traversal(root-left-right)')
    paths = []
    indicator = np.array(indicator.todense())[0]
    for i in range(n_nodes_ptr.shape[0] - 1):
        paths.append(indicator[n_nodes_ptr[i]: n_nodes_ptr[i + 1]])
    return paths


def _orgnize_features(important_features: List[Important_feature]) -> dict[str, int]:
    result = {}
    for feature in important_features:
        key = feature.name
        result[key] = feature.value
    return result


# used to save results of a bacteria and corresponding phage by model
BacteriaPairResult = namedtuple(
    "PairResult", ['phage_file', 'result', 'probability', 'features']
)


def _batch_ouput2stdout(results: dict[str, list[BacteriaPairResult]]):
    '''orgnize batch predicting results and output to stdout, result'll be orgnized into 2 lattices. 
    the first lattice containing the result and probability of each phage-bacteria pair
    the second contains the values of most important features of each phage-bacteria pair
    '''
    # orgnize the first lattice
    lattice1 = []
    headers = ['bacteria_file', 'phage_file', 'result', 'probability']
    for bacteria_file, result_list in results.items():
        for phage_result in result_list:
            crt_result = [bacteria_file, phage_result.phage_file, phage_result.result,
                          str(phage_result.probability * 100)[: 4] + '%']
            lattice1.append(crt_result)

    print('The predicting result of phages and bacterias:')
    init(autoreset=True)
    print(f'{Fore.RED}Notice: if probability is between 40% and 60%,' +
          'the region is fuzzy, predicted result may be incorrect')
    print(tabulate(lattice1, headers=headers,
          tablefmt="fancy_grid", showindex=True))

    # orgnize the second lattice
    print("----------------------------------------------------------------------------------------")
    print("in these prediciting pairs, most important features and their corresponding values are: ")
    print("you might check detailed importance by using \"-fea\" in cli")
    lattice2 = []
    # make headers
    headers = ['bacteria_file', 'phage_file', 'features']
    for bacteria_file, result_list in results.items():
        for phage_result in result_list:
            orgnized_fea_dict = _orgnize_features(phage_result.features)
            crt_result = [bacteria_file,
                          phage_result.phage_file, str(orgnized_fea_dict)]
            lattice2.append(crt_result)
    print(tabulate(lattice2))


def _make_output_files(output_dir: str):
    '''
    make sure the output file does not already exist,
    if already exist, add _ after original name
    Returns:
        csv_out: file descriptor pointing to the output csv
        txt_out:    file descriptor pointing to the output txt
    '''
    csv_output_name = 'PBLKS_result'
    csv_path = os.path.join(output_dir, csv_output_name + ".csv")
    while (os.path.exists(csv_path)):
        csv_output_name += "_"
        csv_path = os.path.join(output_dir, csv_output_name + ".csv")
    txt_output_name = 'PBLKS_result'
    txt_path = os.path.join(output_dir, txt_output_name + ".txt")
    while (os.path.exists(txt_path)):
        txt_output_name += "_"
        txt_path = os.path.join(output_dir, txt_output_name + ".txt")
    csv_out = open(csv_path, 'w')
    txt_out = open(txt_path, 'w')
    return (csv_out, txt_out)


def _make_csv_output(results: dict[str, list[BacteriaPairResult]]) -> dict[str, str]:
    '''
    reorgnize result into csv output form 
    result would be orgnized in {phage file name}: {results(in bacteria line's order)}
    '''
    bac_name0 = list(results.items())[0][0]
    # collect all phage names
    phage_results0 = results[bac_name0]
    phage_names = []
    for result in phage_results0:
        phage_file = result.phage_file
        phage_names.append(phage_file)
    # collect results
    final_results = {phage_name: [] for phage_name in phage_names}
    for bac_name in results.keys():
        phage_results = results[bac_name]
        for result in phage_results:
            final_results[result.phage_file].append(str(result.result))
    # concat into string
    for key, val in final_results.items():
        final_results[key] = ','.join(val)
    return final_results


def _batch_output2file(output_dir: str, results: dict[str, list[BacteriaPairResult]], use_xgb):
    '''orgnize batch predicting results and output to given directory
    result'll be orgnized into 2 files
    csv orgnizes output results(0,1) in lattice like form
    txt provides both result and other important information
    '''
    csv_out, txt_out = _make_output_files(output_dir)
    csv_header = '''
    """1"" represents the query bactrium is predicted as the host of the query phage,""0"" represents the query bactrium is predicted as the nonhost of the query phage",,,,,
    '''
    csv_out.write(csv_header)
    # write bac file names line after header
    bac_names = ""
    for name in results.keys():
        bac_names += "," + name
    csv_out.write(bac_names + "\n")
    result_dict = _make_csv_output(results)
    for file_name, result_str in result_dict.items():
        csv_out.write(file_name + ',' + result_str + "\n")
    csv_out.close()

    # deal with txt output
    if not use_xgb:
        model_type = "RandomForest"
    else:
        model_type = "XGBoost"
    default_param = 'window_length(9000bp)_and_step_size(1800bp)'
    txt_out.write(
        'Bacteria_filename Phage_filename Predicted_result Predicted_probability Default_parameters	Learning_model	Important_features(TOP10)\n')
    for bac_file, phage_results in results.items():
        for phage_result in phage_results:
            phage_file = phage_result.phage_file
            if phage_result.result == 1:
                predict_result = "host"
            else:
                predict_result = "non-host"
            proba = str(phage_result.probability * 100)[:5]
            feature_importance = _orgnize_features(phage_result.features)
            txt_out.write(
                f'{bac_file} {phage_file} {predict_result} {proba}% {default_param} {model_type} {feature_importance}\n')
    txt_out.close()


def predict_in_df(pair_fn:str,
                  phage_folder:str,
                    bacteria_folder:str,
                  use_xgb=False,
                  output_dir=None):
    '''
    Given a dataframe of phage-bacteria pairs, predict the probability of each pair in the dataframe
    Written by Ha Vu
    '''
    pair_df = pd.read_csv(pair_fn, sep='\t', header=0, index_col=None)
    assert 'Phage_ID' in pair_df.columns and 'Host_ID' in pair_df.columns, 'The input dataframe must have columns named Phage_ID and Host_ID'
    model = _get_model(use_xgb)
    result_df = pd.DataFrame(columns=['Phage_ID', 'Host_ID', 'prediction'])
    for index, row in pair_df.iterrows():
        phage_fn = os.path.join(phage_folder, row['Phage_ID'] + '.fasta.gz')
        bac_fn = os.path.join(bacteria_folder, row['Host_ID'] + '.fasta.gz')
        try:
            descriptor = get_descriptor(bac_fn, phage_fn)
            proba = model.predict_proba(descriptor)[0][1]
        except:
            proba= np.nan
        result_df.loc[index] = [row['Phage_ID'], row['Host_ID'], proba]
    if output_dir is not None:
        save_fn = os.path.join(output_dir, 'prediction_result.csv')
        result_df.to_csv(save_fn, sep='\t', index=False)
    return result_df


def predict_in_batch(bacteria_folder: str,
                     phage_folder: str,
                     use_xgb=False,
                     output_dir=None):
    '''predict phage-bacteria interaction in batches, predictes whether each phage in the phage folder
    interacts with each bacteria in the bacteria folder, nested folders are ignored
    result includes the result(0, 1) and probability that the phage interacts with the bacteria, the
    values of most important features
    the output'll be saved into output_dir as 2 files(1 txt, 1csv) or stdout
    Args:
        bac_path: path to the foldr with sequence files of Bacteria(content must be orgnized in fasta format)
        phage_path: path to the foldr with sequence files of Phage
        use_xgb: use XGBoost based model(randomforest by default)
        output_dir: output result to given directory(stdout if None)
    Raises:
        FileNotFoundError: if at least one of the sequence file entered does not exist
        InputFileFormatError: if the content of the sequence file is not orgnized in fasta format
    '''
    if not (os.path.exists(bacteria_folder) and os.path.exists(phage_folder)):
        raise FileNotFoundError(
            'at least one of the sequence folder you entered does not exist :(')
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            raise DirectoryError("output includes 2 files, a folder instead of a file is needed")

    phage_file_names = os.listdir(phage_folder)
    bacteria_file_names = os.listdir(bacteria_folder)
    model = _get_model(use_xgb)

    results = {}  # bacteria_file_name: [BacteriaPairResult of each phage]
    for bacteria_file in bacteria_file_names:
        # if nested folder, ignore
        bacteria_path = os.path.join(bacteria_folder, bacteria_file)
        if os.path.isdir(bacteria_path):
            continue
        # ignore .DS_Store files in mac
        if bacteria_file.count('.DS_Store') != 0:
            continue

        crt_bac_results = []
        for phage_file in phage_file_names:
            # if nested folder, ignore
            phage_path = os.path.join(phage_folder, phage_file)
            if os.path.isdir(phage_file):
                continue
            # ignore .DS_Store files in mac
            if phage_file.count('.DS_Store') != 0:
                continue
            descriptor = get_descriptor(bacteria_path, phage_path)
            proba = model.predict_proba(descriptor)[0][1]
            important_features = _get_important_features(model, 10, descriptor)
            crt_result = BacteriaPairResult(phage_file, int(
                proba > 0.5), proba, important_features)
            crt_bac_results.append(crt_result)
        results[bacteria_file] = crt_bac_results

    if output_dir == None:
        _batch_ouput2stdout(results)
    else:
        _batch_output2file(output_dir, results, use_xgb)
