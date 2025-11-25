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

    # jac_score = metrics.jaccard_score(real_y, pre_y, average='macro')
    print("========external metrics========")
    print("Homogeneity:{:.4f}, V_measure:{:.4f}, MI:{:.4f}, ARI:{:.4f}, NMI:{:.4f}, AMI:{:.4f}".format(h_score, v_score, mi_score, ari_score, nmi_score, ami_score))
    return h_score, v_score, mi_score,ari_score, nmi_score, ami_score

def internal_performance(adata1, adata2, pre_y, spatial, k=10):
    X1 = adata1.obsm['feat']
    X2 = adata2.obsm['feat']

    silhouette_1 = silhouette_score(X1, pre_y)
    silhouette_2 = silhouette_score(X2, pre_y)

    dbi_1 = davies_bouldin_score(X1, pre_y)
    dbi_2 = davies_bouldin_score(X2, pre_y)

    ch_1 = calinski_harabasz_score(X1, pre_y)
    ch_2 = calinski_harabasz_score(X2, pre_y)

    weight = lib.weights.KNN.from_array(spatial, k=k)  # 5近邻权
    moran = Moran(pre_y, weight)
    moran_score = moran.I
    moran_p = moran.p_sim
    print("========internal metrics========")
    print("moran index:{:.4f}, p-value: {}".format(moran_score, moran_p))
    print("omics1 | silhouette:{:.4f}, calinski harabasz:{:.4f}, davies bouldin:{}".format(silhouette_1, dbi_1, ch_1))
    print("omics2 | silhouette:{:.4f}, calinski harabasz:{:.4f}, davies bouldin:{}".format(silhouette_2, dbi_2, ch_2))

    return moran_score, silhouette_1, silhouette_2, dbi_1, dbi_2, ch_1, ch_2