import pandas as pd
import numpy as np
import os
test_prompt='/mnt/efs/fs1/data/metadata/phageScope/test_prompt.txt'
phage_fasta='/mnt/efs/fs1/data/raw/phageScope/phage_fasta/'
host_fasta='/mnt/efs/fs1/data/raw/phageScope/host_fasta/'
output_folder='/mnt/efs/fs1/data/predictions/ha/phageScope/'
num_jobs=8
def break_test_prompt(test_prompt, num_jobs = num_jobs):
    prompt_df = pd.read_csv(test_prompt, sep='\t')
    prompt_df['job'] = np.arange(len(prompt_df)) % num_jobs
    for job in range(num_jobs):
        save_fn = os.path.join(output_folder, f'test_prompt_{job}.txt')
        prompt_df[prompt_df['job'] == job].to_csv(save_fn, sep='\t', index=False, header=True)
    return

break_test_prompt(test_prompt, num_jobs)

rule all:
    input:
        expand(os.path.join(output_folder, 'job{job}', 'prediction_result.csv'), job=range(num_jobs)),
        os.path.join(output_folder, 'prediction_result.csv'),

rule run_PBLKS:
    input:
        os.path.join(output_folder, 'test_prompt_{job}.txt')
    output:
        os.path.join(output_folder, 'job{job}','prediction_result.csv')
    params:
        job_out = os.path.join(output_folder, 'job{job}')
    shell:
        '''
        python PB-LKS.py -p {phage_fasta} -b {host_fasta} -o {params.job_out} --pair_fn {input[0]} -xgb
        '''

rule combine_results:
    input:
        expand(os.path.join(output_folder, 'job{job}', 'prediction_result.csv'), job=range(num_jobs))
    output:
        os.path.join(output_folder, 'prediction_result.csv')
    run:
        dfs = [pd.read_csv(fn, header = 0, index_col=None, sep = '\t') for fn in input]
        df = pd.concat(dfs)
        df.to_csv(output[0], index=False, header=True, sep='\t')