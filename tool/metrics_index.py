from sklearn import metrics
import libpysal
import pysal.lib as lib
from esda.moran import Moran
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def external_performance(real_y, pre_y):
    h_score = metrics.homogeneity_score(real_y, pre_y)
    v_score =  metrics.v_measure_score(real_y, pre_y)
    mi_score = metrics.mutual_info_score(real_y, pre_y)
    
    ari_score = metrics.adjusted_rand_score(real_y, pre_y)
    nmi_score = metrics.normalized_mutual_info_score(real_y, pre_y)
    ami_score = metrics.adjusted_mutual_info_score(real_y, pre_y)

    return h_score, v_score, mi_score,ari_score, nmi_score, ami_score

def internal_performance(adata1, adata2, pre_y, spatial, k=10):
    X1 = adata1.obsm['feat']
    X2 = adata2.obsm['feat']

    weight = lib.weights.KNN.from_array(spatial, k=k)  # 5近邻权
    moran = Moran(pre_y, weight)
    moran_score = moran.I
    moran_p = moran.p_sim

    return moran_score