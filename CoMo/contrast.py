import torch
import torch.nn as nn
import numpy as np

class Contrast(nn.Module):
    def __init__(self, hidden_dim, 
                 tau=0.8, lam=0.5):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lam = lam

        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)  # 对稀疏矩阵的范数计算
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, z_mp, z_sc, pos):
        
        z_proj_mp = self.proj(z_mp)
        z_proj_sc = self.proj(z_sc)

        matrix_mp2sc = self.sim(z_proj_mp, z_proj_sc)
        matrix_sc2mp = matrix_mp2sc.t()
        matrix_mp2sc = matrix_mp2sc/(torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-8)
        matrix_sc2mp = matrix_sc2mp / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8)
        # print("is pos sparse:", pos.is_sparse)
        # print("is matrix_mp2sc sparse:", matrix_mp2sc.is_sparse)
        # print("is matrix_mp2sc sparse:", matrix_mp2sc.is_sparse)
        
        lori_mp = -torch.log(matrix_mp2sc.mul(pos.to_dense()).sum(dim=-1)).mean()
        lori_sc = -torch.log(matrix_sc2mp.mul(pos.to_dense()).sum(dim=-1)).mean()

        return self.lam * lori_mp + (1 - self.lam) * lori_sc



def clustering_loss(out_1, out_2, tau_plus, cluster_num, beta, estimator='hard', temperature=10):
    # neg score
    out = torch.cat([out_1, out_2], dim=1)
    neg = torch.exp(torch.mm(out.t().contiguous(), out) / temperature)

    mask = get_negative_mask(cluster_num).cuda()
    # print("mask:", mask)
    neg = neg.masked_select(mask).view(2 * cluster_num, -1)
    # print("neg:", neg)

    # pos score
    pos = torch.exp(torch.sum(out_1 * out_2, dim=0) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    # negative samples similarity scoring
    if estimator == 'hard':
        N = 2 * cluster_num - 2
        imp = (beta * neg.log()).exp()
        reweight_neg = (imp * neg).sum(dim=-1) / imp.mean(dim=-1)
        Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
        # constrain (optional)
        Ng = torch.clamp(Ng, min=N * np.e ** (-1 / temperature))
    elif estimator == 'easy':
        Ng = neg.sum(dim=-1)
    else:
        raise Exception('Invalid estimator selected. Please use any of [hard, easy]')

    # cluster entropy
    P = torch.mean(out, dim=0)
    H = - (P * torch.log(P)).sum()

    # cluster loss
    loss = (- torch.log(pos / (pos + Ng))).mean() - H
    return loss


def contrastive_loss(out_1, out_2=None, nei_matrix=None,
                     tau_plus=0.05, beta=1, estimator='hard', temperature=10):
    
    # 如果邻接矩阵存在，计算 out_2 = out_1 和 nei_matrix 的乘积
    # print("out_1 shape", out_1.shape)
    if nei_matrix is not None:
        nei_matrix = nei_matrix.to(out_1.device)
        # 使用 torch.sparse.mm 进行稀疏 x 稠密乘法
        out_2 = torch.sparse.mm(nei_matrix, out_1)
    else:
        assert out_2 is not None, "必须提供 out_2 或 nei_matrix"

    batch_size = out_1.size(0)

    # selection 2
    # neg score
    inv_temp = 1.0 / temperature
    out = torch.cat([out_1, out_2], dim=0)
    sim_matrix = torch.matmul(out, out.t()) * inv_temp
    neg = sim_matrix.exp()

    mask = get_negative_mask(batch_size).to(out_1.device)
    neg = neg.masked_fill(~mask, 0.0)

    # pos score
    pos = torch.sum(out_1 * out_2, dim=-1) * inv_temp
    pos = pos.exp()
    pos = torch.cat([pos, pos], dim=0)


    # negative samples similarity scoring
    if estimator == 'hard':
        N = 2 * batch_size - 2
        imp = (beta * neg.log()).exp()
        reweight_neg = (imp * neg).sum(dim=-1) / imp.mean(dim=-1)
        Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
        # constrain (optional)
        Ng = torch.clamp(Ng, min=N * np.e ** (-1 / temperature))
    elif estimator == 'easy':
        Ng = neg.sum(dim=-1)
    else:
        raise Exception('Invalid estimator selected. Please use any of [hard, easy]')

    # contrastive loss
    loss = (- torch.log(pos / (pos + Ng))).mean()
    return loss

def get_negative_mask(size):
    negative_mask = torch.ones((size, 2 * size), dtype=bool)
    for i in range(size):
        negative_mask[i, i] = 0
        negative_mask[i, i + size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask
