import candle
from benchmark_dataset_generator.improve_utils import *
import os, sys
import pandas as pd
from smiles2graph import smiles2graph
import csv
import gzip
import shutil
import pickle
import torch
from torch_geometric.data import Data

'''
# Running notes
# please note, the versions of dependencies are complicate, it is hard to make all things compatible.
# 1. Activate conda environment "TGSA_pre" for preprocessing code (requires PyTorch >= 1.9.0 because of DGL),
# 2. While, active "TGSA" for model training and inference (PyTorch < 1.7.1 because of torch-geometric)
'''

ftp_origin = "https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/csa_data"
x_dir = os.path.join(ftp_origin, "x_data")
y_dir = os.path.join(ftp_origin, "y_data")

mut_fn = "cancer_mutation.txt"
mut_cnt_fn = "cancer_mutation_count.txt"
exp_fn = "cancer_gene_expression.txt"
cn_fn = "cancer_copy_number.txt"
resp_fn = "response.txt"
drug_smile_fn = "drug_SMILES.txt"

data_root_dir = "/infodev1/non-phi-projects/junjiang/TGSA/benchmark_dataset_generator/csa_data"


###########################################################################################
# download data
###########################################################################################
# Argonne's dataset
candle.file_utils.get_file(fname=resp_fn, origin=os.path.join(y_dir, resp_fn),
                           datadir=os.path.join(data_root_dir, "raw_data"),
                           cache_subdir="y_data")
candle.file_utils.get_file(fname=mut_fn, origin=os.path.join(x_dir, mut_fn),
                           datadir=os.path.join(data_root_dir, "raw_data"),
                           cache_subdir="x_data")
candle.file_utils.get_file(fname=exp_fn, origin=os.path.join(x_dir, exp_fn),
                           datadir=os.path.join(data_root_dir, "raw_data"),
                           cache_subdir="x_data")
candle.file_utils.get_file(fname=cn_fn, origin=os.path.join(x_dir, cn_fn),
                           datadir=os.path.join(data_root_dir, "raw_data"),
                           cache_subdir="x_data")
candle.file_utils.get_file(fname=drug_smile_fn, origin=os.path.join(x_dir, drug_smile_fn),
                           datadir=os.path.join(data_root_dir, "raw_data"),
                           cache_subdir="x_data")

# Extra data
STRING_info = "https://stringdb-static.org/download/protein.info.v11.5/9606.protein.info.v11.5.txt.gz"
candle.file_utils.get_file(fname='9606.protein.info.v11.5.txt.gz', origin=STRING_info,
                           datadir=os.path.join(data_root_dir, "raw_data"),
                           cache_subdir="x_data")

STRING_links_details = "https://stringdb-static.org/download/protein.links.detailed.v11.5/9606.protein.links.detailed.v11.5.txt.gz"
candle.file_utils.get_file(fname='9606.protein.links.detailed.v11.5.txt.gz', origin=STRING_links_details,
                           datadir=os.path.join(data_root_dir, "raw_data"),
                           cache_subdir="x_data")


##########################################################################################
def save_drug2graph(dg_smiles_df, save_to):
    drug_dict = {}
    for i in range(len(dg_smiles_df)):
        drug_dict[dg_smiles_df.iloc[i, 0]] = smiles2graph(dg_smiles_df.iloc[i, 1])
    if ".npy" in save_to:
        np.save(save_to, drug_dict)
    elif '.pkl' in save_to:
        with open(save_to, 'wb') as f:
            pickle.dump(drug_dict, f)
    return drug_dict


def rreplace(s, old, new):
    li = s.rsplit(old, 1)  # Split only once
    return new.join(li)


def unzip_gz_file(fn):
    with gzip.open(fn, 'rb') as f_in:
        new_fn = rreplace(fn, ".gz", "")
        with open(new_fn, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return new_fn


def ensp_to_hugo_map(fn):
    with open(fn, "r") as csv_file:
        next(csv_file)  # Skip first line
        csv_reader = csv.reader(csv_file, delimiter='\t')
        ensp_map = {row[0]: row[1] for row in csv_reader if row[0] != ""}
    return ensp_map


# create mapping of the IDs
def hugo_to_ncbi_map(fn):
    with open(fn) as csv_file:
        next(csv_file)  # Skip first line
        csv_reader = csv.reader(csv_file, delimiter='\t')
        hugo_map = {row[0]: int(row[1]) for row in csv_reader if row[1] != ""}

    return hugo_map



###########################################################################################
# save drug_feature_graph.npy
###########################################################################################
drug_smile = load_smiles_data()
dr_df = load_single_drug_response_data(source="y_data")
save_to_fn = os.path.join(data_root_dir, "drug_feature_graph.pkl")

if not os.path.exists(save_to_fn):
    drug_dict = save_drug2graph(drug_smile, save_to_fn)
else:
    if ".pkl" == os.path.splitext(save_to_fn)[1]:
        with open(save_to_fn, 'rb') as ffp:
            drug_dict = pickle.load(ffp)
    elif ".npy" == os.path.splitext(save_to_fn)[1]:
        drug_dict = np.load(save_to_fn, allow_pickle=True).item()
    else:
        Exception("does not support this file format for drug graph")
# for debug
merge_1 = pd.merge(drug_smile, dr_df, how='inner', on='improve_chem_id')
selected_drugs = list(set(merge_1["improve_chem_id"]))
print("Number of drugs: %d" % len(selected_drugs))
improve_sample_id = list(set(dr_df['improve_sample_id']))
print("Number of cell lines: %d" % len(improve_sample_id))
###########################################################################################
# 1. select related genes based on protein-protein interaction (PPI) scores
# 2. create graph to represent cell lines, each node is a gene:
#       a. connections were defined by PPI related genes
#       b. node features were expression level, copy number variation and mutation
###########################################################################################
thresh = 0.95
# save_to_dir = "/infodev1/non-phi-projects/junjiang/TGSA/benchmark_dataset_generator/csa_data"
cell_dict_save_to = os.path.join(data_root_dir, 'cell_feature_all.pkl')
edge_index_fn = os.path.join(data_root_dir, 'edge_index_PPI_{}.npy'.format(thresh))
selected_gene_fn = os.path.join(data_root_dir, 'selected_gen_PPI_{}.pkl'.format(thresh))
if os.path.exists(cell_dict_save_to) and os.path.exists(edge_index_fn) and os.path.exists(selected_gene_fn):
    fp = open(cell_dict_save_to, "rb")
    cell_dict = pickle.load(fp)
    fp.close()
    fp = open(selected_gene_fn, "rb")
    selected_gene = pickle.load(fp)
    fp.close()
    edge_index = np.load(edge_index_fn)
else:
    ##############################################################
    gene_fn = os.path.join(data_root_dir, "raw_data", "enterez_NCBI_to_hugo_gene_symbol_march_2019.txt")

    protein_info_gz_fn = os.path.join(data_root_dir, "raw_data", "x_data", "9606.protein.info.v11.5.txt.gz")
    protein_links_gz_fn = os.path.join(data_root_dir, "raw_data", "x_data", "9606.protein.links.detailed.v11.5.txt.gz")
    protein_info_fn = protein_info_gz_fn.replace(".txt.gz", ".txt")
    protein_links_fn = protein_links_gz_fn.replace(".txt.gz", ".txt")
    if os.path.exists(protein_info_fn) and os.path.exists(protein_links_fn):
        pass
    else:
        protein_info_fn = unzip_gz_file(protein_info_gz_fn)
        protein_links_fn = unzip_gz_file(protein_links_gz_fn)
    ########################## use protein-protein iteraction to filter gene######################
    edges = pd.read_csv(protein_links_fn, sep=' ')
    selected_edges_filter1 = edges[edges['combined_score'] > (thresh * 1000)]
    edge_list = selected_edges_filter1[["protein1", "protein2"]].values.tolist()
    hugo_map = hugo_to_ncbi_map(gene_fn)
    ensp_map = ensp_to_hugo_map(protein_info_fn)
    selected_gene_tmp = set()
    for i in edge_list:
        selected_gene_tmp.add(ensp_map[i[0]])
        selected_gene_tmp.add(ensp_map[i[1]])


    # level_map = {"Ensembl": 0, "Entrez": 1, "Gene_Symbol": 2}
    print("\t\treading gene expression data")
    exp_df = pd.read_csv(improve_globals.gene_expression_file_path, sep="\t", index_col=0, header=2, low_memory=False)
    print("\t\treading gene copy number variation data")
    cn_df = pd.read_csv(improve_globals.copy_number_file_path, sep="\t", index_col=0, header=1, low_memory=False)
    print("\t\treading gene mutation data")
    mu_df = pd.read_csv(improve_globals.gene_mutation_file_path, sep="\t", index_col=0, header=0, low_memory=False)

    exp_df_keys = set(exp_df.keys())
    cn_df_keys = set(cn_df.keys())
    mu_df_keys = set(mu_df.keys())
    selected_gene = mu_df_keys.intersection(cn_df_keys.intersection(exp_df_keys.intersection(selected_gene_tmp)))
    gene_list = list(selected_gene)
    ############################## save selected gene and gene features for cell lines ########################
    print("Number of selected gene: %d" % len(gene_list))
    fp = open(selected_gene_fn, "wb")
    pickle.dump(selected_gene, fp)  # save selected gene names
    fp.close()

    cell_exp_df = exp_df[selected_gene]
    cell_exp = cell_exp_df.iloc[0:]
    cell_cn_df = cn_df[selected_gene]
    cell_cn = cell_cn_df.iloc[1:]

    cell_mu_df = mu_df[selected_gene]
    # cell_mu = cell_mu_df.iloc[11:]
    cell_mu = cell_mu_df.iloc[0:]

    cell_line_ids = list(cell_exp.index)
    cell_dict = {}   # key: cell_line_name; value: features of selected genes
    for cl_df in enumerate(cell_line_ids):
        cl = list(cl_df)[1]
        exp = list(cell_exp.loc[cl])
        cn = list(cell_cn.loc[cl])
        mu = list(cell_mu.loc[cl])
        exp = [float(i) for i in exp]
        cn = [float(i) for i in cn]
        mu = [float(i) for i in mu]
        cell_dict[cl] = Data(x=torch.tensor(np.array([exp, cn, mu]).T, dtype=torch.float))

    fp = open(cell_dict_save_to, "wb")  # save gene features for cell lines
    pickle.dump(cell_dict, fp)
    fp.close()

    ############################## create graph edges between genes ########################
    edge_list = [[ensp_map[edge[0]], ensp_map[edge[1]]] for edge in edge_list if
                 edge[0] in ensp_map.keys() and edge[1] in ensp_map.keys()]

    edge_index = []   # connections between genes. The values are index of selected genes.
    for i in edge_list:
        if (i[0] in gene_list) & (i[1] in gene_list):
            edge_index.append((gene_list.index(i[0]), gene_list.index(i[1])))
            edge_index.append((gene_list.index(i[1]), gene_list.index(i[0])))
    edge_index = list(set(edge_index))
    edge_index = np.array(edge_index, dtype=np.int64).T

    # save edge_index
    print("Average degree of gene graph for cell lines: %f (combined score threshold:%f)" % (len(edge_index[0]) / len(gene_list), thresh))
    np.save(edge_index_fn, edge_index)

print("done")
