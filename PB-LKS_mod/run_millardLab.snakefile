import pandas as pd
import numpy as np
phage_folder = '/u/home/h/havu73/project-ernst/data/hackathonBio/millarLab/phage_trial'
host_folder = '/u/home/h/havu73/project-ernst/data/hackathonBio/millarLab/host_trial'
output_fn = './trial'

rule all:
    input:
        output_fn,

rule run_PB_LKS:
    input:
    output:
        output
    shell:
        '''
        python PB-LKS.py -p {phage_folder} -b {host_folder} -ba -o {output_fn}
        '''