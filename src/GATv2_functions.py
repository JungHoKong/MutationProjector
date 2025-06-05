# GATv2_onehop : one hop convolution
# GATv2block : integrate multiple networks
import numpy as np
from collections import defaultdict
import scipy.stats as stat
import pandas as pd
import sklearn
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.ensemble import *
from sklearn.linear_model import *
from itertools import *
import os, time, sys, random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from collections import defaultdict

# GATv2_onehop
class GATv2_onehop(nn.Module):
    def __init__(self, num_node_features, num_heads, self_loop, GAT_dropout=0):
        super(GATv2_onehop, self).__init__()
        self.num_node_features = num_node_features
        
        # GATv2 layer
        self.conv1 = GATv2Conv(self.num_node_features, self.num_node_features, heads=num_heads, concat=False, add_self_loops=self_loop, dropout=GAT_dropout)


    def forward(self, x, network_edge, batch_size, return_attention_weights):
        # GATv2 layer
        if return_attention_weights == False:
            x = self.conv1(x, network_edge)
            x = x.view(batch_size, -1, self.num_node_features)
            return x
        else:
            x, x2 = self.conv1(x, network_edge, return_attention_weights=return_attention_weights)
            x = x.view(batch_size, -1, self.num_node_features)
            edge_index, attention_weights = x2
            return x, edge_index, attention_weights

        
        
# GATv2block
class GATv2block(nn.Module):
    def __init__(self, num_genes, num_features, network_edges, num_GATblock, num_heads, dropout_p, cuda_device, d_ff=100, self_loop=True):
        super(GATv2block, self).__init__()
        self.num_genes = num_genes
        self.num_features = num_features
        self.network_edges = network_edges
        self.num_GATblock = num_GATblock
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.cuda_device = cuda_device
        self.d_ff = d_ff
        self.self_loop = self_loop
        
        # GATv2 layers
        GAT_layers = []
        self.GATidx = defaultdict(list)
        count = 0
        for i in range(self.num_GATblock):
            for j in range(len(self.network_edges)):
                GAT_layers.append(GATv2_onehop(self.num_features, self.num_heads, self.self_loop, GAT_dropout=self.dropout_p))
                self.GATidx['i'].append(i)
                self.GATidx['j'].append(j)
                self.GATidx['idx'].append(count)
                count+=1
        self.GAT_layers = nn.Sequential(*GAT_layers)
        self.GATidx = pd.DataFrame(self.GATidx)
        
        
        # Linear transformation
        self.linear_layers = nn.ModuleList([nn.Linear(num_features * len(self.network_edges), num_features).cuda(self.cuda_device) for _ in range(self.num_genes)])
        self.FF_layer1 = nn.ModuleList([nn.Linear(self.num_features, self.d_ff).cuda(self.cuda_device) for _ in range(self.num_genes)])
        self.FF_layer2 = nn.ModuleList([nn.Linear(self.d_ff, self.num_features).cuda(self.cuda_device) for _ in range(self.num_genes)])

        # dropout
        self.dropout = nn.Dropout(p=self.dropout_p)
        
        # layer norm
        self.layer_norm1 = nn.ModuleList([nn.LayerNorm(self.num_features).cuda(self.cuda_device) for _ in range(self.num_GATblock)])
        self.layer_norm2 = nn.ModuleList([nn.LayerNorm(self.num_features).cuda(self.cuda_device) for _ in range(self.num_GATblock)])
        
        
    def forward(self, X_input, return_attention_weights):
        X_original = X_input.clone()
        X_original = X_original.cuda(self.cuda_device)
        batch_size = X_input.shape[0]
        
        # attention weights (output)
        out_edges, out_att_weights = {}, {}
        
        # GAT block
        for i in range(self.num_GATblock):
            # run GAT layer
            for j in range(len(self.network_edges)):
                # first GAT layer
                if i == 0:
                    X_train = X_original.clone()
                else:
                    X_train = X.clone()
                # data loader
                data_list = [Data(x=X_train[k], edge_index=self.network_edges[j]) for k in range(X_train.shape[0])]
                loader = torch_geometric.loader.DataLoader(data_list, batch_size=batch_size)
                # GAT layer
                GATidx = self.GATidx.loc[self.GATidx['i']==i,:].loc[self.GATidx['j']==j,:]['idx'].tolist()[0]
                GATlayer = self.GAT_layers[GATidx].cuda(self.cuda_device)
                
                for batch in loader:
                    # does not return attention weights
                    if return_attention_weights == False:
                        x = GATlayer(batch.x.cuda(self.cuda_device), batch.edge_index.cuda(self.cuda_device), X_train.shape[0], False)
                        
                    # return attention weights
                    else:
                        x, edge_index, attention_weights = GATlayer(batch.x.cuda(self.cuda_device), batch.edge_index.cuda(self.cuda_device),  X_train.shape[0], return_attention_weights)
                        out_edges['%s_%s'%(i, j)] = edge_index
                        out_att_weights['%s_%s'%(i, j)] = attention_weights
                        
                # concat
                if j == 0:
                    x_cat = x
                else:
                    x_cat = torch.cat((x_cat, x), dim=2)
                
            # linear layer for GAT outputs
            X_gat = []
            for gene_i in range(self.num_genes):
                out_gene = self.linear_layers[gene_i](x_cat[:,gene_i,:])
                X_gat.append(out_gene)
            X_gat = torch.stack(X_gat, dim=1)
            
            # Add residual connections and normalize
            X = self.layer_norm1[i](X_train + self.dropout(X_gat))

            # Feed Forward linear layer
            X_ff = []
            for gene_i in range(self.num_genes):
                out_gene = self.FF_layer1[gene_i](X[:,gene_i,:])
                out_gene = torch.relu(out_gene)
                out_gene = self.FF_layer2[gene_i](out_gene)
                X_ff.append(out_gene)
            X_ff = torch.stack(X_ff, dim=1)

            # Add residual connections and normalize
            X = self.layer_norm2[i](X + self.dropout(X_ff))

            
        if return_attention_weights == False:
            return X
        else:
            return X, out_edges, out_att_weights
        