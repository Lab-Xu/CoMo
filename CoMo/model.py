import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import collections
from .pyWNN import pyWNN
import numpy as np

class Encoder_overall(Module):
      
    """\
    Overall encoder.

    Parameters
    ----------
    dim_in_feat_omics1 : int
        Dimension of input features for omics1.
    dim_in_feat_omics2 : int
        Dimension of input features for omics2. 
    dim_out_feat_omics1 : int
        Dimension of latent representation for omics1.
    dim_out_feat_omics2 : int
        Dimension of latent representation for omics2, which is the same as omics1.
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.    

    Returns
    -------
    results: a dictionary including representations and modality weights.

    """
     
    def __init__(self, dim_in_feat_omics1, dim_in_feat_omics2, 
                 hidden_dim_list, 
                 dropout=0.5, act=F.relu, 
                #  modality_num=2, disc_depth=1,
                 cluster_num=None,
                 use_adj_feature=True,
                 use_cross_attention=False,
                 use_cluster_head=False,
                 use_prejection_head=False,
                 use_WNN=False,
                 w1=0.5, 
                 w2=0.5,
                 device=None,
                #  use_disc=False,
                 ):
        super(Encoder_overall, self).__init__()
        self.dim_in_feat_omics1 = dim_in_feat_omics1
        self.dim_in_feat_omics2 = dim_in_feat_omics2
        self.P = len(hidden_dim_list)
        self.en_hid_dim_omics1 = [dim_in_feat_omics1] + hidden_dim_list
        self.de_hid_dim_omics1 = self.en_hid_dim_omics1[::-1]

        self.en_hid_dim_omics2 = [dim_in_feat_omics2] + hidden_dim_list
        self.de_hid_dim_omics2 = self.en_hid_dim_omics2[::-1]

        self.dim_out_feat_omics1 = hidden_dim_list[-1]
        self.dim_out_feat_omics2 = hidden_dim_list[-1]
        self.dropout = dropout
        self.act = act
        self.use_adj_feature = use_adj_feature
        self.use_cross_attention = use_cross_attention
        self.use_cluster_head = use_cluster_head
        self.latent_dim = hidden_dim_list[-1]
        self.use_prejection_head = use_prejection_head
        self.use_WNN = use_WNN
        self.w1=w1
        self.w2=w2
        self.device = device
        # self.use_disc = use_disc
        
        self.encoder_omics1 = nn.ModuleList([GCN(self.en_hid_dim_omics1[i], self.en_hid_dim_omics1[i+1]) for i in range(self.P)])
        self.decoder_omics1 = nn.ModuleList([GCN(self.de_hid_dim_omics1[i], self.de_hid_dim_omics1[i+1]) for i in range(self.P)])

        self.encoder_omics2 = nn.ModuleList([GCN(self.en_hid_dim_omics2[i], self.en_hid_dim_omics2[i+1]) for i in range(self.P)])
        self.decoder_omics2 = nn.ModuleList([GCN(self.de_hid_dim_omics2[i], self.de_hid_dim_omics2[i+1]) for i in range(self.P)])
        # print("encoder_omics1:", self.encoder_omics1)
        
        self.atten_omics1 = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics1)
        self.atten_omics2 = AttentionLayer(self.dim_out_feat_omics2, self.dim_out_feat_omics2)
        self.atten_cross = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics2)
        self.cross_attention_block = CrossAttentionLayer(self.latent_dim)

        self.cluster_head_1 = nn.Sequential(
                                        nn.BatchNorm1d(self.latent_dim),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(self.latent_dim, self.latent_dim),
                                        nn.BatchNorm1d(self.latent_dim),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(self.latent_dim, cluster_num),
                                        nn.Softmax(dim=1),
                                    )

        self.cluster_head_2 = nn.Sequential(
                                        nn.BatchNorm1d(self.latent_dim),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(self.latent_dim, self.latent_dim),
                                        nn.BatchNorm1d(self.latent_dim),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(self.latent_dim, cluster_num),
                                        nn.Softmax(dim=1),
                                    )

        self.projection_head = nn.Sequential(nn.Linear(self.latent_dim, 64, bias=False), 
                                             nn.BatchNorm1d(64), 
                                             nn.ReLU(inplace=True), 
                                             nn.Dropout(p=dropout),
                                             nn.Linear(64, 32, bias=True))
        
    def forward(self, adata, current_epoch,
                features_omics1, features_omics2, 
                adj_spatial_omics1, adj_feature_omics1, 
                adj_spatial_omics2, adj_feature_omics2,
                wnn_epoch=100, wnn_neighbors=20,
                w=1):
        # print(f"w1:{self.w1}, w2:{self.w2}")
        
        if self.use_adj_feature:
            # graph1
            emb_latent_spatial_omics1 = features_omics1
            emb_latent_spatial_omics2 = features_omics2
            emb_latent_feature_omics1 = features_omics1
            emb_latent_feature_omics2 = features_omics2

            for i in range(self.P):
                emb_latent_spatial_omics1 = self.encoder_omics1[i](emb_latent_spatial_omics1, adj_spatial_omics1)
                emb_latent_spatial_omics2 = self.encoder_omics2[i](emb_latent_spatial_omics2, adj_spatial_omics2)
                # graph2
                emb_latent_feature_omics1 = self.encoder_omics1[i](emb_latent_feature_omics1, adj_feature_omics1)
                emb_latent_feature_omics2 = self.encoder_omics2[i](emb_latent_feature_omics2, adj_feature_omics2)
            # within-modality attention aggregation layer
            emb_latent_omics1, alpha_omics1 = self.atten_omics1(emb_latent_spatial_omics1, emb_latent_feature_omics1)
            emb_latent_omics2, alpha_omics2 = self.atten_omics2(emb_latent_spatial_omics2, emb_latent_feature_omics2)
        else:
            emb_latent_omics1 = features_omics1
            emb_latent_omics2 = features_omics2
            for i in range(self.P):
                emb_latent_omics1 = self.encoder_omics1[i](emb_latent_omics1, adj_spatial_omics1)
                emb_latent_omics2 = self.encoder_omics2[i](emb_latent_omics2, adj_spatial_omics2)
            alpha_omics1 = torch.tensor(0)
            alpha_omics2 = torch.tensor(0)

        latent_concat = torch.cat([emb_latent_omics1, emb_latent_omics2])

        if self.use_cross_attention:
            emb_cross_omics1  = self.cross_attention_block(emb_latent_omics1.unsqueeze(1), 
                                                            emb_latent_omics2.unsqueeze(1)).squeeze(1)
            emb_cross_omics2 = self.cross_attention_block(emb_latent_omics2.unsqueeze(1), 
                                                            emb_latent_omics1.unsqueeze(1)).squeeze(1)
            # emb_latent_combined, alpha_omics_1_2 = self.atten_cross(emb_cross_omics1, emb_cross_omics2)
        else:
            # emb_latent_combined, alpha_omics_1_2 = self.atten_cross(emb_latent_omics1, emb_latent_omics2)
            emb_cross_omics1 = torch.tensor(0)
            emb_cross_omics2 = torch.tensor(0)
        
        if self.use_WNN:

            if self.use_cross_attention:
                x1 = emb_cross_omics1
                x2 = emb_cross_omics2
            else:
                x1 = emb_latent_omics1
                x2 = emb_latent_omics2

            if current_epoch % wnn_epoch == 0:
                w = 1
                print("=====Run WNN=====")
            else:
                w = 0
            
            if w==1:
                pc1 = x1.detach().cpu().numpy()
                pc2 = x2.detach().cpu().numpy()
                # print("pc1:", pc1)
                # print("pc2:", pc2)
                adata.obsm['Omics1_PCA'] = pc1
                adata.obsm['Omics2_PCA'] = pc2
                WNNobj = pyWNN(adata, reps=['Omics1_PCA', 'Omics2_PCA'], 
                            npcs=[x1.shape[1], x2.shape[1]], 
                            n_neighbors=wnn_neighbors, seed=2025)
                adata = WNNobj.compute_wnn(adata)
                ww = adata.obsm['Weights']
                ww = ww.astype(np.float32)
                w1 = torch.reshape(torch.from_numpy(ww[:,0]),(-1,1)).to(self.device)
                w2 = torch.reshape(torch.from_numpy(ww[:,1]),(-1,1)).to(self.device)
            else:
                w1 = torch.tensor(self.w1, device=self.device)
                w2 = torch.tensor(self.w2, device=self.device)
            emb_latent_combined = x1 * w1 + x2 * w2
            alpha_omics_1_2 = torch.tensor(0)
        else:
            if self.use_cross_attention:
                emb_latent_combined, alpha_omics_1_2 = self.atten_cross(emb_cross_omics1, 
                                                                        emb_cross_omics2)
            else:
                emb_latent_combined, alpha_omics_1_2 = self.atten_cross(emb_latent_omics1, 
                                                                        emb_latent_omics2)


        if self.use_cluster_head:
            cluster_omics1 = self.cluster_head_1(emb_latent_omics1)
            cluster_omics2 = self.cluster_head_2(emb_latent_omics2)
        else:
            cluster_omics1 = None
            cluster_omics2 = None
        
        if self.use_prejection_head:
            projection = self.projection_head(emb_latent_combined)
            projection = F.normalize(projection, dim=-1)
        else:
            projection = torch.tensor(0)

        # if self.use_disc:
        #     y_disc = self.disc(emb_latent_combined)
        # else:
        #     y_disc = torch.tensor(0)
    

        # reverse the integrated representation back into the original expression space with modality-specific decoder
        emb_recon_omics1 = emb_latent_combined
        emb_recon_omics2 = emb_latent_combined
        for i in range(self.P):
            emb_recon_omics1 = self.decoder_omics1[i](emb_recon_omics1, adj_spatial_omics1)
            emb_recon_omics2 = self.decoder_omics2[i](emb_recon_omics2, adj_spatial_omics2)
        
        
        results = {'emb_latent_omics1':emb_latent_omics1,
                   'emb_latent_omics2':emb_latent_omics2,
                   'emb_latent_combined':emb_latent_combined,
                   'emb_recon_omics1':emb_recon_omics1,
                   'emb_recon_omics2':emb_recon_omics2,
                #    'emb_latent_omics1_across_recon':emb_latent_omics1_across_recon,
                #    'emb_latent_omics2_across_recon':emb_latent_omics2_across_recon,
                   'alpha_omics1':alpha_omics1,
                   'alpha_omics2':alpha_omics2,
                   'alpha':alpha_omics_1_2,
                   'emb_cross_omics1':emb_cross_omics1,
                   'emb_cross_omics2':emb_cross_omics2,
                   'cluster_omics1':cluster_omics1,
                   'cluster_omics2':cluster_omics2,
                   'latent_concat':latent_concat,
                   'projection':projection,
                #    'y_disc':y_disc,
                   }
        
        return results


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.414)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj):
        seq_fts = self.fc(seq)
        out = torch.spmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)

class Encoder(Module): 
    
    """\
    Modality-specific GNN encoder.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features. 
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.    

    Returns
    -------
    Latent representation.

    """
    
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)
        
        return x
    
class Decoder(Module):
    
    """\
    Modality-specific GNN decoder.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features. 
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.    

    Returns
    -------
    Reconstructed representation.

    """
    
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act
        
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)
        
        return x                  
     

class AttentionLayer(Module):
    
    """\
    Attention layer.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features.     

    Returns
    -------
    Aggregated representations and modality weights.

    """
    
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)
        
    def forward(self, emb1, emb2):
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        self.emb = torch.cat(emb, dim=1)
        
        self.v = torch.tanh(torch.matmul(self.emb, self.w_omega)) # torch.tanh, F.tanh
        self.vu=  torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6, dim=1) # , dim=1
        
        emb_combined = torch.matmul(torch.transpose(self.emb,1,2), torch.unsqueeze(self.alpha, -1))
    
        return torch.squeeze(emb_combined), self.alpha

# self or cross attention
class Attention(nn.Module):
    def __init__(self, dim, n_heads=8, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        # self.recons_tensor = Recons_tensor(2)
        
        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):

        # print("x[0] shape:", x[0].shape)
        n_samples, n_tokens, dim = x[0].shape
        if dim != self.dim:
            raise ValueError

        n_tokens_en = n_tokens
        q = self.q_linear(x[0]).reshape(n_samples, n_tokens, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_linear(x[1]).reshape(n_samples, n_tokens_en, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_linear(x[2]).reshape(n_samples, n_tokens_en, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches + 1)
        dp = (q @ k_t) * self.scale  # (n_samples, n_heads, n_patches, n_patches)
        dp = -1 * dp
        attn = dp.softmax(dim=-1)  # (n_samples, n_heads, n_patches, n_patches)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches +1, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2)  # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches + 1, dim)

        x = self.proj(weighted_avg)  # (n_samples, n_patches + 1, dim)
        x = self.proj_drop(x)  # (n_samples, n_patches + 1, dim)
        
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        # x = x + 1e-6
        x = self.fc1(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.act(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.fc2(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)

        return x

class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim,
        )

    def forward(self, x):
        x_ = [self.norm1(_x) for _x in x]
        out = x[2] + self.attn(x_)
        out = out + self.mlp(self.norm2(out))
        out = [x_[0], out, out]
        
        return out

class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, depth=2, n_heads=8,
                 mlp_ratio=4., qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.pos_drop = nn.Dropout(p=p)
        self.blocks = nn.ModuleList(
            [
                Block(dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p=p, attn_p=attn_p)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, query, key):
        key = self.pos_drop(key)
        value = key
        x = [query, key, value]
        for block in self.blocks:
            x = block(x)
            x[2] = self.norm(x[2])
        x_self = x[2]
        return x_self



# class CrossAttentionLayer(Module):
#     def __init__(self, latent_dim):
#         super(CrossAttentionLayer, self).__init__()
#         self.query_proj = nn.Linear(latent_dim, latent_dim)
#         self.key_proj = nn.Linear(latent_dim, latent_dim)
#         self.value_proj = nn.Linear(latent_dim, latent_dim)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, query, key, value):
#         queries = self.query_proj(query)
#         keys = self.key_proj(key)
#         values = self.value_proj(value)
#         attention_weights = self.softmax(torch.bmm(queries, keys.transpose(1, 2)))
#         attended_output = torch.bmm(attention_weights, values)
#         return attended_output
    
class Discriminator(torch.nn.Sequential):

    r"""
    Modality discriminator
    """

    def __init__(
            self, in_features=64, modality_num=2,
            h_depth=1, h_dim=32,
            dropout=0.2
    ):

        od = collections.OrderedDict()
        ptr_dim = in_features
        for layer in range(h_depth):
            od[f"linear_{layer}"] = torch.nn.Linear(ptr_dim, h_dim)
            od[f"act_{layer}"] = torch.nn.LeakyReLU(negative_slope=0.2)
            od[f"dropout_{layer}"] = torch.nn.Dropout(p=dropout)
            ptr_dim = h_dim
        od["pred"] = torch.nn.Linear(ptr_dim, modality_num)
        super().__init__(od)

    def forward(self, embed):
        x = super().forward(embed)
        y = F.softmax(x, dim=1)
        return y