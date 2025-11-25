import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment


def compute_cross_entropy_loss(P, Q):
    """
    Compute the cross-entropy loss L_p based on the given formula.
    
    Parameters:
    P (list of torch.Tensor): List of target probability matrices [P^1, P^2, ..., P^M],
        each P^m of shape [N, K] where P_{ij}^m is the target probability.
    Q (list of torch.Tensor): List of predicted probability matrices [Q^1, Q^2, ..., Q^M],
        each Q^m of shape [N, K] where Q_{ij}^m is the predicted probability.
    
    Returns:
    torch.Tensor: The total cross-entropy loss L_p.
    """
    loss = 0.0
    M = len(P)  # Number of modalities
    
    for m in range(M):
        # Ensure P^m and Q^m are on the same device and have valid probabilities
        # GUI: Change
        P_m = P[m].clamp(min=1e-10, max=1.0)  # Avoid log(0)
        Q_m = Q[m].clamp(min=1e-10, max=1.0)  # Avoid log(0)
        
        # Compute cross-entropy loss for modality m
        modality_loss = -torch.sum(P_m * torch.log(Q_m))
        loss += modality_loss
    
    return loss



def compute_high_level_cluster(global_features, q_predictions, num_clusters):

    # 1. 拼接全局特征
    N = global_features.shape[0]
    # 2. K-means聚类 (Eq.11)
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    cluster_assignments = kmeans.fit_predict(global_features.cpu().detach().numpy())
    centroids = torch.tensor(kmeans.cluster_centers_, device=global_features.device)  # [K, M*H]
    
    # 3. 最大匹配伪标签生成 (Eq.12-13)
    refined_pseudo_labels = []
    cross_entropy_loss = 0
    for m in range(len(q_predictions)):
        # 获取当前模态的预测标签
        hard_q = torch.argmax(q_predictions[m], dim=1).cpu().numpy()  # [N]
        
        # 构建代价矩阵 (Eq.13)
        cost_matrix = np.zeros((num_clusters, num_clusters))
        for i in range(num_clusters):
            for j in range(num_clusters):
                # 统计同时属于聚类i(预测)和j(kmeans)的样本数
                mask = (hard_q == i) & (cluster_assignments == j)
                cost_matrix[i, j] = -np.sum(mask)  # 负号因匈牙利算法求解最小化
        
        # 匈牙利算法求解最优匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assignment_matrix = np.zeros((num_clusters, num_clusters))
        assignment_matrix[row_ind, col_ind] = 1
        
        # 生成调整后的伪标签
        adjusted_labels = np.zeros((N, num_clusters))
        for j in range(num_clusters):
            adjusted_labels[cluster_assignments == j] = assignment_matrix[:, j]
        
        refined_pseudo_labels.append(torch.tensor(adjusted_labels, device=global_features.device))

    return refined_pseudo_labels, q_predictions
