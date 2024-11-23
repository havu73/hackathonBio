import pandas as pd
import os
import json
ONE_KB= 1000
DFT_READ_LENGTH = 150
DFT_TOSS_DISTANCE = 300
DFT_BUFFER_JUNCTION=9  # number of bf before the intron start site that we will consider a read to be a junction read.
# for a read length READ_LENGTH to be considered a junction read, the start of the read must be within (READ_LENGTH-BUFFER_JUNCTION-1, BUFFER_JUNCTION) bp of ahead the intron start site
DEFAULT_SEED = 9999
DFT_ETA_VALUE=200
DFT_INSERTSIZE_MIN=200
DFT_INSERTSIZE_MAX=300
DFT_UNIF_FRAG_LEN = 250
DEFAULT_SIM_FEAT_LEN = 5000 # length of one feature in the simulation, default value

def save_json_file(json_obj, path):
    create_folder_for_file(path)
    f = open(path, 'w')
    json.dump(json_obj, f, indent=4)
    return

def check_colnames_in_df(columns_to_check, df):
	"""
	:param columns_to_check: list of columns that we want to check to be present in a dataframe df
	:param df: the dataframe
	:return: if there are missing columns in columns to check, we will need to terminate the program
	"""
	assert set(columns_to_check).issubset(df.columns), ('Columns specified by users are NOT ALL present '
														'in the input dataframes: {}').format(columns_to_check)

def make_dir(directory):
	try:
		os.makedirs(directory)
	except:
		pass



def check_file_exist(fn):
	if not os.path.isfile(fn):
		print ( "File: " + fn + " DOES NOT EXISTS")
		exit(1)
	return 

def check_dir_exist(fn):
	if not os.path.isdir(fn):
		print ( "Directory: " + fn + " DOES NOT EXISTS")
		exit(1)
	return 
	
def create_folder_for_file(fn):
	last_slash_index = fn.rfind('/')
	if last_slash_index != -1: # path contains folder
		make_dir(fn[:last_slash_index])
	return 

def get_command_line_integer(arg):
	try: 
		arg = int(arg)
		return arg
	except:
		print ( "Integer: " + str(arg) + " IS NOT VALID")
		exit(1)

def get_command_line_float(arg):
	try: 
		arg = float(arg)
		return arg
	except:
		print ( "Integer: " + str(arg) + " IS NOT VALID")
		exit(1)

def get_enrichment_df (enrichment_fn): # enrichment_fn follows the format of ChromHMM OverlapEnrichment's format
	enrichment_df = pd.read_csv(enrichment_fn, sep = "\t")
	# rename the org_enrichment_df so that it's easier to work with
	enrichment_df =	enrichment_df.rename(columns = {"state (Emission order)": "state", "Genome %": "percent_in_genome"})
	return enrichment_df

def get_non_coding_enrichment_df (non_coding_enrichment_fn):
	nc_enrichment_df = pd.read_csv(non_coding_enrichment_fn, sep = '\t')
	if len(nc_enrichment_df.columns) != 3:
		print ( "Number of columns in a non_coding_enrichment_fn should be 3. The provided file has " + str(len(nc_enrichment_df.columns)) + " columns.")
		print ( "Exiting, from ChromHMM_untilities_common_functions_helper.py")
		exit(1)
	# Now, we know that the nc_enrichment_df has exactly 3 columns
	# change the column names
	nc_enrichment_df.columns = ["state", "percent_in_genome", "non_coding"]
	return (nc_enrichment_df)