import candle
from benchmark_dataset_generator.improve_utils import *
import os, sys
import pandas as pd
from smiles2graph import smiles2graph


# please note: the version compatibility is complicate.
# activate TGSA_pre for preprocessing code (requires PyTorch >= 1.9.0 because of DGL),
# while, active TGSA for model training and inference (PyTorch < 1.7.1 because of torch-geometric)


# refer to Rohan's repository

ftp_origin = "https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/csa_data"
x_dir = os.path.join(ftp_origin, "x_data")
y_dir = os.path.join(ftp_origin, "y_data")

mut_fn = "cancer_mutation.txt"
mut_cnt_fn = "cancer_mutation_count.txt"
exp_fn = "cancer_gene_expression.txt"
cn_fn = "cancer_copy_number.txt"
resp_fn = "response.txt"
drug_smile_fn = "drug_SMILES.txt"

# params = {"model_name": "csa_data_dir", "experiment_id": "1", "run_id": "1"}
# list_dir = candle.file_utils.directory_tree_from_parameters(params, commonroot="./")

def save_drug2graph(dg_smiles_df):
    drug_dict = {}
    for i in range(len(dg_smiles_df)):
        drug_dict[dg_smiles_df.iloc[i, 0]] = smiles2graph(dg_smiles_df.iloc[i, 1])
    np.save('./benchmark_dataset_generator/csa_data/drug_feature_graph.npy', drug_dict)
    return drug_dict

###########################################################################################
# download data
candle.file_utils.get_file(fname=resp_fn, origin=os.path.join(y_dir, resp_fn),
                           datadir="./benchmark_dataset_generator/csa_data/raw_data",
                           cache_subdir="y_data")
candle.file_utils.get_file(fname=mut_fn, origin=os.path.join(x_dir, mut_fn),
                           datadir="./benchmark_dataset_generator/csa_data/raw_data",
                           cache_subdir="x_data")
candle.file_utils.get_file(fname=exp_fn, origin=os.path.join(x_dir, exp_fn),
                           datadir="./benchmark_dataset_generator/csa_data/raw_data",
                           cache_subdir="x_data")
candle.file_utils.get_file(fname=cn_fn, origin=os.path.join(x_dir, cn_fn),
                           datadir="./benchmark_dataset_generator/csa_data/raw_data",
                           cache_subdir="x_data")
candle.file_utils.get_file(fname=drug_smile_fn, origin=os.path.join(x_dir, drug_smile_fn),
                           datadir="./benchmark_dataset_generator/csa_data/raw_data",
                           cache_subdir="x_data")

STRING_links = "https://stringdb-static.org/download/protein.links.v11.5.txt.gz"
candle.file_utils.get_file(fname='protein.links.v11.5.txt.gz', origin=STRING_links,
                           datadir="./benchmark_dataset_generator/csa_data/raw_data",
                           cache_subdir="x_data")

STRING_links_details = "https://stringdb-static.org/download/protein.links.detailed.v11.5.txt.gz"
candle.file_utils.get_file(fname='protein.links.detailed.v11.5.txt.gz', origin=STRING_links_details,
                           datadir="./benchmark_dataset_generator/csa_data/raw_data",
                           cache_subdir="x_data")

##########################################################################################
def save_drug2graph(dg_smiles_df, save_to):
    drug_dict = {}
    for i in range(len(dg_smiles_df)):
        drug_dict[dg_smiles_df.iloc[i, 0]] = smiles2graph(dg_smiles_df.iloc[i, 1])
    np.save(save_to, drug_dict)
    return drug_dict


drug_smile = load_smiles_data()
dr_df = load_single_drug_response_data(source="y_data")
save_to_fn = './benchmark_dataset_generator/csa_data/drug_feature_graph.npy'

# save drug_feature_graph.npy
if not os.path.exists(save_to_fn):
    save_drug2graph(drug_smile, save_to_fn)

merge_1 = pd.merge(drug_smile, dr_df, how='inner', on='improve_chem_id')
# selected drugs
selected_drugs = list(set(merge_1["improve_chem_id"]))

improve_sample_id = list(set(dr_df['improve_sample_id']))
###########################################################################################
# TODO: save cell_feature_all.npy
exp_df = load_gene_expression_data(gene_system_identifier="Gene_Symbol")
# cn_df = load_copy_number_data(gene_system_identifier="Entrez")
# mu_df = load_mutation_data(gene_system_identifier="Entrez")

exp_df = pd.read_csv(improve_globals.gene_expression_file_path, sep="\t", nrows=100, index_col=0, skiprows=0)
cn_df = pd.read_csv(improve_globals.copy_number_file_path, sep="\t", nrows=100, index_col=0, skiprows=0)
mu_df = pd.read_csv(improve_globals.gene_mutation_file_path, sep="\t", nrows=100, index_col=0, skiprows=0)

###########################################################################################


merge_2 = pd.merge(merge_1, dr_df, how='inner', on='improve_chem_id')



print("done")
