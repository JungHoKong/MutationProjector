import pandas as pd
import numpy as np
from collections import defaultdict
import scipy.stats as stat
from itertools import *
import os, time, sys, random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader



class GAT_attention_weights(nn.Module):

    def __init__(self, model, X, X_special_tokens=[]):
        '''
        Inputs
        model : pretrained neural net
        X : input features (N_sample, N_tokenized_features)
        X_special_tokens: input for special tokens (tmb, aneuploidy, mutational signatures). (N_sample, N_signatures). Provide Numpy array or tensor. 
        '''
        super(GAT_attention_weights, self).__init__()
        self.model = model
        self.X = X
        self.X_special_tokens = X_special_tokens

        
                 
    def return_attention_weight_matrix(self):
        '''
        Outputs
        out_weights, GAT_layer
        out_weights : (N_sample, N_GAT_layer, N_gene (Source), N_gene (Target))
        GAT_layer : GAT layer info ('EncoderUnit_Network')
        '''
        # number of samples / genes
        num_samples = self.X.shape[0]
        num_genes = self.X.shape[1]
        if len(self.X_special_tokens) > 0:
            num_genes += self.X_special_tokens.shape[1]

        # attention weights
        self.model.eval()
        with torch.no_grad():
            pred, masked_positions, attention_weights, edge_indices, gene_emb = self.model(self.X, self.X_special_tokens, return_attention_weights=True)

        # initialize attention weights (with nan values! Nan values will be substitued will real values)
        out_weights = np.full( (num_samples, len(edge_indices), num_genes, num_genes), np.nan )

        # replace with real attention weights
        GAT_layer = []
        for GATlayer_idx, GATlayer in enumerate(list(edge_indices.keys())):
            GAT_layer.append(GATlayer)
            # edges and attention weights
            tmp_edges = edge_indices[GATlayer]
            tmp_weights = attention_weights[GATlayer]
            # iterate
            for edge_idx in range(tmp_edges.shape[1]):
                node1, node2 = tmp_edges.detach().cpu()[:,edge_idx]
                node1, node2 = node1.item(), node2.item()
                assert node1//num_genes == node2//num_genes, f"Potential error in preparing edge indices. node1={node1}, node2={node2}"

                # sample index
                sample_idx = node1//num_genes
                # source node, target node
                source_node = node1 - sample_idx * num_genes
                target_node = node2 - sample_idx * num_genes

                # attention weight
                W = tmp_weights[edge_idx].item()
                # replace nan value with real attention weight
                out_weights[sample_idx][GATlayer_idx][source_node][target_node] = W

        return out_weights, GAT_layer

    
    def average_weights(self):
        '''
        element-wise average across all attention matrices. 
        nan values are ignored when averaging values.
        In this average attention weight matrix, A(i, j) represent how much attention from gene i (source) was paid to gene j (target).
        '''
        import warnings
        warnings.simplefilter('ignore', category=RuntimeWarning)
        out_weights, GAT_layer = self.return_attention_weight_matrix()
        out = np.nanmean(out_weights, axis=1)
        return out

    
    def max_weights(self):
        '''
        element-wise max across all attention matrices. 
        nan values are ignored when averaging values.
        In this max attention weight matrix, A(i, j) represent how much "maximum" attention from gene i (source) was paid to gene j (target).
        '''
        import warnings
        warnings.simplefilter('ignore', category=RuntimeWarning)
        out_weights, GAT_layer = self.return_attention_weight_matrix()
        out = np.nanmax(out_weights, axis=1)
        return out
    
    
    
    def attention_sum_vector(self, summarize_by='max'):
        '''
        Attention weight-based gene importance.
            Computes the sum of columns of the {average, max} attention weight matrix (A(i,j)).
        '''
        if summarize_by == 'average':
            summarized_weights = self.average_weights()
        elif summarize_by == 'max':
            summarized_weights = self.max_weights()
        out = np.nansum(summarized_weights, axis=1) #2)
        return out

    
    def attention_mean_vector(self):
        '''
        Attention weight-based gene importance.
        Computes the mean of columns of the average attention weight matrix (A(i,j)).
        '''
        import warnings
        warnings.simplefilter('ignore', category=RuntimeWarning)
        avg_weights = self.average_weights()
        out = np.nanmean(avg_weights, axis=1) #2)
        return out
    