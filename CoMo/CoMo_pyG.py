import torch
from tqdm import tqdm
import torch.nn.functional as F
from .model import Encoder_overall
from .preprocess import adjacent_matrix_preprocessing
from .contrast import contrastive_loss, clustering_loss
from .clustering_distribution import compute_cross_entropy_loss
import numpy as np
import time

class Train_CoMo:
    def __init__(self, 
        data,
        modality_class,
        device= torch.device('cpu'),
        random_seed = 2025,
        hidden_dim_list = [64],
        learning_rate=0.0001,
        weight_decay=0.00,
        epochs=600, 
        wnn_epoch=100,
        wnn_neighbors=20,
        dim_output=64,
        use_adj_feature=True,
        use_cross_attention=True,
        cluster_num=None,
        use_cluster_head=True,
        cluster_loss_w = 0,
        tau_plus=0.5, 
        beta=1,
        estimator='hard',
        temperature=10,
        use_contrastive_loss=True,
        contrast_loss_w = 0,
        use_WNN=True,
        stage_weights=None,
        stage_epochs=None,
        modality_num=2,
        ):

        self.data = data.copy()
        self.device = device
        self.random_seed = random_seed
        self.hidden_dim_list = hidden_dim_list
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.dim_output = dim_output
        self.use_adj_feature = use_adj_feature
        self.use_cross_attention = use_cross_attention
        self.cluster_num = cluster_num
        self.use_cluster_head = use_cluster_head
        self.cluster_loss_w = cluster_loss_w
        self.tau_plus = tau_plus
        self.beta = beta
        self.estimator = estimator
        self.temperature = temperature
        self.use_contrastive_loss = use_contrastive_loss
        self.use_WNN = use_WNN
        self.contrast_loss_w = contrast_loss_w
        self.wnn_epoch = wnn_epoch
        self.wnn_neighbors = wnn_neighbors
        
        self.modality_num = modality_num
        self.modality_class_onehot = torch.tensor(np.array(modality_class)).long().to(self.device)
        # F.one_hot(torch.tensor(modality_class)).float().cuda()
        
        # adj
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2)
        self.adj_contrast = self.adj['adj_contrast'].to(self.device)
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to(self.device)
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to(self.device)
        self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to(self.device)
        self.adj_feature_omics2 = self.adj['adj_feature_omics2'].to(self.device)
        # self.dense_adj = self.data['dense_adj'].to(self.device)
        print('self.adj_spatial_omics1 len:', len(self.adj_spatial_omics1))

        self.pos_omics1 = self.adj['adj_spatial_omics1']
        new_values = torch.ones(self.pos_omics1._values().size())
        self.pos_omics1 = torch.sparse_coo_tensor(self.pos_omics1._indices(), new_values, self.pos_omics1.size()).to(self.device)

        self.pos_omics2 = self.adj['adj_spatial_omics2']
        new_values = torch.ones(self.pos_omics2._values().size())
        self.pos_omics2 = torch.sparse_coo_tensor(self.pos_omics2._indices(), new_values, self.pos_omics2.size()).to(self.device)
        
        # feature
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)
        self.n_cell_omics1 = self.adata_omics1.n_obs
        self.n_cell_omics2 = self.adata_omics2.n_obs
        
        # dimension of input feature
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.hidden_dim_list = hidden_dim_list
        # self.dim_output1 = self.dim_output
        # self.dim_output2 = self.dim_output
        
        # Dynamic weight scheduling for stages
        if stage_weights:
            self.stage_weights = stage_weights
        else:
            self.stage_weights = {
                'recon': [1.0, 5.0, 0.0, 0.0, 0.0],  # Stage 1: reconstruction
                'contrast': [1.0, 5.0, 1.0, 5.0, 0.0],  # Stage 2: reduce recon and cluster (for alignment), increase contrast
            }
        self.current_stage = 'recon'
        if stage_epochs:
            self.stage_epochs = stage_epochs
        else:
            self.stage_epochs = {'recon': 300, 'contrast': 600}
        self.current_epoch = 0
    
    def train(self, step_by_step_train=False,
              w1=0.5, w2=0.5):

        self.model = Encoder_overall(self.dim_input1, self.dim_input2, self.hidden_dim_list,
                                     use_adj_feature=self.use_adj_feature,
                                     use_cross_attention=self.use_cross_attention,
                                     use_cluster_head=self.use_cluster_head,
                                     cluster_num=self.cluster_num,
                                     use_prejection_head=self.use_contrastive_loss,
                                     use_WNN=self.use_WNN,
                                     w1=w1, w2=w2,
                                     device=self.device).to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), self.learning_rate, 
                                        weight_decay=self.weight_decay) # Adam
        
        if step_by_step_train:
            loss = self.step_by_step_train_model(total_epochs=self.epochs,
                                                 wnn_epoch=self.wnn_epoch,
                                                 wnn_neighbors=self.wnn_neighbors)
        else:
            pass

        print("Model training finished!\n")
    
        with torch.no_grad():
          self.model.eval()
          epoch=100
          results = self.model(self.adata_omics1, epoch,
                               self.features_omics1, self.features_omics2, 
                               self.adj_spatial_omics1, self.adj_feature_omics1, 
                               self.adj_spatial_omics2, self.adj_feature_omics2,)

        emb_omics1 = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)
        emb_omics2 = F.normalize(results['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)
        
        output = {'emb_latent_omics1': emb_omics1.detach().cpu().numpy(),
                  'emb_latent_omics2': emb_omics2.detach().cpu().numpy(),
                  'CoMo': emb_combined.detach().cpu().numpy(),
                  'alpha_omics1': results['alpha_omics1'].detach().cpu().numpy(),
                  'alpha_omics2': results['alpha_omics2'].detach().cpu().numpy(),
                  'alpha': results['alpha'].detach().cpu().numpy(),
                  'emb_latent_omics1_origin': results['emb_latent_omics1'].detach().cpu().numpy(),
                  'emb_latent_omics2_origin': results['emb_latent_omics2'].detach().cpu().numpy(),
                  'emb_latent_combined':results['emb_latent_combined'].detach().cpu().numpy(),
                  }
        
        return output
    
    def _update_stage(self):
        """Update training stage based on epoch."""
        if self.current_epoch < self.stage_epochs['recon']:
            self.current_stage = 'recon'
        elif self.current_epoch < self.stage_epochs['recon'] + self.stage_epochs['contrast']:
            self.current_stage = 'contrast'

    def step_by_step_train_model(self, total_epochs=600, 
                                 wnn_epoch=100, wnn_neighbors=20):
        print("step by step train model")
        self.model.train()
        loss_history = {'recon': [],'contrast': []}
        

        for epoch in tqdm(range(total_epochs)):
            self.current_epoch = epoch
            self.optimizer.zero_grad()
            results = self.model(self.adata_omics1, epoch,
                                 self.features_omics1, self.features_omics2, 
                                 self.adj_spatial_omics1, self.adj_feature_omics1, 
                                 self.adj_spatial_omics2, self.adj_feature_omics2,
                                 wnn_epoch=wnn_epoch, wnn_neighbors=wnn_neighbors)

            # Stage management
            self._update_stage()
            weights = self.stage_weights[self.current_stage]

            # reconstruction loss
            loss_recon_omics1 = F.mse_loss(self.features_omics1, results['emb_recon_omics1'])
            loss_recon_omics2 = F.mse_loss(self.features_omics2, results['emb_recon_omics2'])
            loss_recon = weights[0] * loss_recon_omics1 + weights[1] * loss_recon_omics2

            loss_cluster = 0.0
            if self.use_cluster_head and self.current_stage in ['contrast']:
                cluster_omics1 = results['cluster_omics1']
                cluster_omics2 = results['cluster_omics2']
                loss_cluster = clustering_loss(cluster_omics1, cluster_omics2, 
                                            cluster_num=self.cluster_num, 
                                            tau_plus=self.tau_plus, beta=self.beta,
                                            temperature=self.temperature)
                loss_cluster *= weights[2]
            
            loss_contrast = 0.0
            if self.use_contrastive_loss and self.current_stage in ['contrast']:

                loss_contrast = contrastive_loss(results['projection'], nei_matrix=self.adj_contrast,
                                                tau_plus=self.tau_plus, beta=self.beta,
                                                estimator=self.estimator,
                                                temperature=self.temperature,
                                                )
                loss_contrast *= weights[3]

            # Total loss
            loss = loss_recon + loss_cluster + loss_contrast
            loss.backward()
            self.optimizer.step()

            # Logging
            loss_history['recon'].append(loss_recon)
            # loss_history['cluster'].append(loss_cluster)
            loss_history['contrast'].append(loss_contrast)
            
            if epoch % 100 == 0:
                print('current_stage:', self.current_stage)
                print("weights:", weights)

        return loss

    def train_discriminator_model(self):
        self.model_d.train()
        self.optimizer_d.zero_grad()
        results = self.model(self.features_omics1, self.features_omics2, self.adj_spatial_omics1, self.adj_feature_omics1, self.adj_spatial_omics2, self.adj_feature_omics2)
        y_disc = self.model_d(results['latent_concat'])
        loss_disc = F.cross_entropy(y_disc, self.modality_class_onehot)
        loss_disc.backward()
        self.optimizer_d.step()
        return loss_disc



    
        
    
    
