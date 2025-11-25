from CoMo.preprocess import construct_neighbor_graph
from CoMo.CoMo_pyG import Train_CoMo
from CoMo.utils import clustering
import os
import torch
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from tool.metrics_index import external_performance, internal_performance
from CoMo.preprocess import fix_seed
from CoMo.preprocess import clr_normalize_each_cell, pca
from collections import Counter

def cluster_purity_metric(listA, listB, num=1):
    # 提取listA中1对应的cluster index
    selected_clusters = [cluster for a, cluster in zip(listA, listB) if a == num]
    
    # 处理无1的特殊情况
    if not selected_clusters:
        return 0.0  # 或根据需求返回np.nan/抛出异常
    
    # 计算最大类别占比（纯度）
    cluster_counts = Counter(selected_clusters)
    max_count = max(cluster_counts.values())
    purity = max_count / len(selected_clusters)
    print(f"purity:{purity}")
    return purity


def run_algorithms(omics_list, 
                   RNA_data_type = '10x',
                   have_anno=False,
                    n_clusters=6,
                    n_neighbors=3,
                    n_contrast_neighbors=5,
                    hidden_dim_list=[64],
                    epochs=600,
                    wnn_epoch=100,
                    wnn_neighbors=20,
                    step_by_step_train=False,
                    stage_weights=None,
                    stage_epochs=1200,
                    w1=0.5, w2=0.5,
                    use_adj_feature=True,
                    use_cross_attention=False,
                    use_cluster_head=False,
                    cluster_loss_w=1,
                    tau_plus = 0.5,
                    estimator='hard',
                    use_contrastive_loss = False,
                    temperature=10,
                    contrast_loss_w=1,
                    use_WNN=False,
                    cluster_method='mclust',
                    refinement=True,
                    n_neigh_refine=5,
                    random_seed=2026,
                   ):
    # print(f"w1:{w1}, w2:{w2}")
    print(f"========seed:{random_seed}========")
    fix_seed(random_seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.environ['R_HOME'] = 'D:\software\R-4.0.3'

    # set modality_class
    class_index = 0
    modality_class = []
    for omics in omics_list:
        modality_class = modality_class + [class_index]*omics.X.shape[0]
        class_index += 1
    
    adata_omics1, adata_omics2 = omics_list[0], omics_list[1]
    # print("adata_omics1:", adata_omics1)
    # print("adata_omics2:", adata_omics2)
    
    # Construct Graph Data
    data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=RNA_data_type, 
                                    n_neighbors=n_neighbors, n_contrast_neighbors=n_contrast_neighbors)
    
    # Train Model
    model = Train_CoMo(data, modality_class=modality_class,
                            device=device, 
                            hidden_dim_list=hidden_dim_list,
                            epochs=epochs, 
                            wnn_epoch=wnn_epoch,
                            wnn_neighbors=wnn_neighbors,
                            use_adj_feature=use_adj_feature,
                            use_cross_attention=use_cross_attention,
                            use_cluster_head=use_cluster_head,
                            cluster_loss_w=cluster_loss_w,
                            cluster_num=n_clusters,
                            tau_plus=tau_plus,
                            estimator=estimator,
                            use_contrastive_loss=use_contrastive_loss,
                            contrast_loss_w=contrast_loss_w,
                            use_WNN=use_WNN,
                            temperature=temperature,
                            stage_weights=stage_weights,
                            stage_epochs=stage_epochs,
                            )

    output = model.train(step_by_step_train=step_by_step_train,
                         w1=w1, w2=w2,)
    adata = adata_omics1.copy()
    adata.obsm['emb_latent_omics1'] = output['emb_latent_omics1'].copy()
    adata.obsm['emb_latent_omics2'] = output['emb_latent_omics2'].copy()
    adata.obsm['CoMo'] = output['CoMo'].copy()

    # Clustering
    clustering(adata, key='CoMo', add_key=cluster_method, n_clusters=n_clusters, 
               method=cluster_method, use_pca=True,
               refinement=refinement, n_neigh_refine=n_neigh_refine)

    # Metric clustering results
    pre_y = adata.obs[cluster_method] # omics1_y, CoMo
    # print("pre_y:", pre_y)
    spatial = adata.obsm['spatial']
    internal_metric = internal_performance(adata_omics1, adata_omics2, pre_y, spatial)

    if have_anno:
        real_y = adata.obs['ground_truth']
        external_metric = external_performance(real_y, pre_y)
    else:
        external_metric = None,None,None,None,None,None

    if refinement:
        refine_pre_y = adata.obs['refine_CoMo']
        # print("refine_pre_y:", refine_pre_y)
        refine_internal_metric = internal_performance(adata_omics1, adata_omics2, refine_pre_y, spatial)

        if have_anno:
            refine_external_metric = external_performance(real_y, refine_pre_y)
        else:
            refine_external_metric = None,None,None,None,None,None
    else:
        refine_internal_metric = None,None,None,None,None,None,None
        refine_external_metric = None,None,None,None,None,None

    return adata, external_metric, internal_metric, refine_external_metric, refine_internal_metric

if __name__ == "__main__":
    # read data
    file_fold = '../data/Human_Lymph/' #please replace 'file_fold' with the download path

    adata_omics1 = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
    adata_omics2 = sc.read_h5ad(file_fold + 'adata_ADT.h5ad')

    adata_omics1.var_names_make_unique()
    adata_omics2.var_names_make_unique()
    anno = pd.read_csv(file_fold + 'annotation.csv')
    anno = anno.set_index('Barcode')
    # adata_omics1.obs_names
    adata_omics1.obs['ground_truth'] = anno.loc[adata_omics1.obs_names, 'manual-anno']

    sc.pp.filter_genes(adata_omics1, min_cells=10)
    sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata_omics1, target_sum=1e4)
    sc.pp.log1p(adata_omics1)
    sc.pp.scale(adata_omics1)

    adata_omics1_high =  adata_omics1[:, adata_omics1.var['highly_variable']]
    adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=adata_omics2.n_vars-1)

    # Protein
    adata_omics2 = clr_normalize_each_cell(adata_omics2)
    sc.pp.scale(adata_omics2)
    adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars-1)

    omics_list = [adata_omics1, adata_omics2]
    adata, external_metric, internal_metric = run_algorithms(omics_list)
    
