import os
import PBLKS
import argparse
from colorama import init, Fore

# CLI tool of PBLKS

print("")
print(" +--------------------------------------------+")
print(" | |---\  |----\       |       |  /   /----\  |")
print(" | |    | |     |      |       | /   |        |")
print(" | |___/  |____/   --  |       |/\    \----\  |")
print(" | |      |     \      |       |  \         | |")
print(" | |      |_____/      |_____  |   \  \____/  |")
print(" +--------------------------------------------+")
print("")

parser = argparse.ArgumentParser()
parser.add_argument(
    '-p', '--phage', help='path to your phage sequence file(in fasta format)')
parser.add_argument(
    '-b', '--bac', help='path to your bacteria sequence file(in fasta format)')
parser.add_argument('--pair_fn', help='path to your pair file')
parser.add_argument(
    '-e', '--example', help='run model with example input and quit', action='store_true')

parser.add_argument('-xgb', '--xgboost',
                    help='run prediction with xgboost(default model is based on RandomForest)', action='store_true')
parser.add_argument('-fea', '--feature',
                    help='show 10 features of most importance and exit', action='store_true')
parser.add_argument(
    '-ba', '--batch', help='run prediction in batch, -p,-b should be folder if set True', action='store_true')
parser.add_argument(
    '-o', '--output', help='the folder you want to save batch prediction results')
parser.add_argument('-d', '--detail',
                    help='print detailed prediction results(single prediction), including important features and decision path', action='store_true')


args = parser.parse_args()

PBLKS.predict_in_df(args.pair_fn, args.phage, args.bac, args.xgboost, args.output)
# # run model with example inputs
# if args.example:
#     print('running pb-lks with example input files')
#     bac_path = 'Example/Bacteria_genome.fasta'
#     phage_path = 'Example/Phage_genome.fasta'
#     print(
#         f'phage sequences is from file: {phage_path}, genome of Aeromonas phage')
#     print(
#         f'bacteria sequences is from file: {bac_path}, genome of Escherichia coli STEC_94C')
#     print('parsing phage sequences')
#     (pred_y, prob_y) = PBLKS.predict(bac_path, phage_path, args.xgboost)
#     if pred_y == 1:
#         print('The prediction result of PB-LKS is: 1.(The queried bacteria is the host of the query phage.)')
#     else:
#         print('The prediction result of PB-LKS is: 0.(The queried bacteria is not the host of the query phage.)')
#     print(
#         f'Predicted probability that the bacteria is the host of the phage is: {prob_y * 100}%')
#     print('The prediction is over.')
#     exit(0)
#
# # print 10 most important features and corresponding importances
# if args.feature:
#     if args.xgboost:
#         feature_dict = PBLKS.show_feature_importance(use_xgboost=True)
#     else:
#         feature_dict = PBLKS.show_feature_importance(use_xgboost=False)
#     print('Top 10 most important kmers and their importances are:')
#     init()
#     colors = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA,
#               Fore.CYAN, Fore.WHITE, Fore.LIGHTBLUE_EX, Fore.LIGHTGREEN_EX, Fore.LIGHTYELLOW_EX]
#     for kmer, color in zip(feature_dict, colors):
#         print(f'{color}{kmer}: {str(feature_dict[kmer]*100)[: 4]}%')
#         bar = '#' * int(feature_dict[kmer] * 2000)
#         print(f'{color}{bar}')
#     exit(0)
#
#
# phage_path = args.phage
# bac_path = args.bac
# if phage_path is None or bac_path is None:
#     print('please provide both a phage and a bacteria file path/ folder')
#     exit(0)
#
# if not args.batch:
#     print(f'phage sequences is from file: {phage_path}')
#     print(f'bacteria sequences is from file: {bac_path}')
#
#     # check whether both file exists
#     file_exists = False
#     while not file_exists:
#         file_exists = os.path.exists(phage_path)
#         if not file_exists:
#             print('The path of phage sequences you provided does not exist :(')
#             phage_path = input(
#                 're-enter a valid phage file path or press CTRL+C(Windows&Linux)/Command+C(MacOS) to exit:')
#     file_exists = False
#     while not file_exists:
#         file_exists = os.path.exists(bac_path)
#         if not file_exists:
#             print('The path of bacteria sequences you provided does not exist :(')
#             bac_path = input(
#                 're-enter a valid bacteria file path or press CTRL+C(Windows&Linux)/Command+C(MacOS) to exit:')
#
#     print('start predicting')
#     use_xgb = args.xgboost
#     need_detail = args.detail
#
#     if need_detail:
#         (pred_y, prob_y, fea_dict) = PBLKS.predict(
#             bac_path, phage_path, use_xgb, need_detail)
#     else:
#         (pred_y, prob_y) = PBLKS.predict(bac_path, phage_path, use_xgb)
#     if pred_y == 1:
#         print('The prediction of PB-LKS is: 1.')
#         print('The queried bacteria is the host of the queried phage.')
#     else:
#         print('The prediction of PB-LKS is: 0.')
#         print('The queried bacteria is not the host of the queried phage.')
#     print(
#         f'Predicted probability that the bacteria is the host of the phage is: {prob_y * 100}%')
#     print('finished predicting.')
#     # print 10 most important features and corresponding values
#     # also show decision path
#     if need_detail:
#         print('the most important features and their values are :')
#         for kmer, fea in fea_dict.items():
#             print(f'{kmer}: {fea}')
#         decision_paths = PBLKS.show_decision_path(
#             bac_path, phage_path, use_xgb)
#         for idx, path in enumerate(decision_paths):
#             print(f'decision path of tree{idx} is:{path} ')
# # predict in batch
# else:
#     if not (os.path.isdir(phage_path) and os.path.isdir(bac_path)):
#         print("when predicting in batch, phage and a bacteria input is expected to be folders.")
#         exit(0)
#     if not (os.path.exists(phage_path) and os.path.exists(phage_path)):
#         print("at least one of your input file does not exist.")
#         exit(0)
#
#     print('start predicting')
#     use_xgb = args.xgboost
#     output_dir = args.output
#     PBLKS.predict_in_df(args.pair_fn, phage_path, bac_path, use_xgb, output_dir)
#     # PBLKS.predict_in_batch(bac_path, phage_path, use_xgb, output_dir)
#     print('finished predicting.')