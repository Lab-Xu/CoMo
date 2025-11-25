import copy
import scanpy as sc
import numpy as np
import time
import tool.file_tool as ft
import os
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import main
from CoMo.preprocess import clr_normalize_each_cell, pca, lsi
import matplotlib.pyplot as plt
from CoMo.preprocess import fix_seed

dataset_name = 'Human_Breast_Cancer'
RNA_data_type = 'other'
omics_type2 = 'ATAC'
clusters_num = 15

stage_weights = {
                'recon': [1.0, 5.0, 0.0, 0.0, 0.0],  # Stage 1: reconstruction
                'contrast': [1.0, 5.0, 1, 5.0, 0.0],  # Stage 2: reduce recon and cluster, increase contrast
            }
stage_epochs = {'recon': 200, 'contrast': 400}

# read data
file_fold = f'../data/{dataset_name}/' #please replace 'file_fold' with the download path
protein_h5_file = 'CytAssist_FFPE_Protein_Expression_Human_Breast_Cancer' + '_raw_probe_bc_matrix.h5'
rna_h5_file =  'CytAssist_FFPE_Protein_Expression_Human_Breast_Cancer' + '_filtered_feature_bc_matrix.h5'
adata_rna = sc.read_visium(path=file_fold, count_file=rna_h5_file)
adata_rna.obsm['spatial'] = adata_rna.obsm['spatial'].astype(int)

full_protein_path = os.path.join(file_fold, protein_h5_file)
adata_protein = sc.read_10x_h5(full_protein_path, gex_only=False) 


common_barcodes = adata_rna.obs_names.intersection(adata_protein.obs_names)  # 获取公共 barcodes
adata_rna_filtered = adata_rna[common_barcodes, :].copy()
adata_protein_filtered = adata_protein[common_barcodes, :].copy()
adata_protein_filtered.obsm['spatial'] = adata_rna_filtered.obsm['spatial']
adata_omics1 = adata_rna_filtered.copy()
adata_omics2 = adata_protein_filtered.copy()
fix_seed(seed=2025)

# preprocess
n_comps = 1000
# RNA
sc.pp.filter_genes(adata_omics1, min_cells=10)
sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata_omics1, target_sum=1e4)
sc.pp.log1p(adata_omics1)
sc.pp.scale(adata_omics1)

adata_omics1_high =  adata_omics1[:, adata_omics1.var['highly_variable']]
adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=n_comps-1)

# Protein
adata_omics2 = clr_normalize_each_cell(adata_omics2)
sc.pp.scale(adata_omics2)
adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=n_comps-1)

omics_list = [adata_omics1, adata_omics2]

# run main model
adata, exter_metric, inter_metric, refine_exter, refine_inter = main.run_algorithms(omics_list, 
                                                            RNA_data_type=RNA_data_type,
                                                            have_anno=False,
                                                            n_clusters=clusters_num,
                                                            n_neighbors=5,
                                                            step_by_step_train=True,
                                                            stage_weights=stage_weights,
                                                            stage_epochs=stage_epochs,
                                                            random_seed=2026)
