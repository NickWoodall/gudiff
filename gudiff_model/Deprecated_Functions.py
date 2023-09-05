import torch
import numpy as np
import util.npose_util as nu
import os
import pathlib
import dgl
from dgl import backend as F
from torch.utils.data import random_split, DataLoader, Dataset
from typing import Dict
from torch import Tensor
from dgl import DGLGraph
from torch import nn
from torch import einsum
import time

from se3_transformer.model.basis import get_basis, update_basis_with_fused
from se3_transformer.model.transformer import Sequential, SE3Transformer
from se3_transformer.model.transformer_topk import SE3Transformer_topK
from se3_transformer.model.FAPE_Loss import FAPE_loss, Qs2Rs, normQ
from se3_transformer.model.layers.attentiontopK import AttentionBlockSE3
from se3_transformer.model.layers.linear import LinearSE3
from se3_transformer.model.layers.convolution import ConvSE3, ConvSE3FuseLevel
from se3_transformer.model.layers.norm import NormSE3
from se3_transformer.model.layers.pooling import GPooling, Latent_Unpool, Unpool_Layer
from se3_transformer.runtime.utils import str2bool, to_cuda
from se3_transformer.model.fiber import Fiber
from se3_transformer.model.transformer import get_populated_edge_features



def get_midpoint(ep_in):
    """Get midpoint, of each batched set of points"""
    
    #calculate midpoint
    midpoint = ep_in.sum(axis=1)/np.repeat(ep_in.shape[1], ep_in.shape[2])
    
    return midpoint

def normalize_points(input_xyz, print_dist=False):
    
    #broadcast to distance matrix [Batch, M, R3] to [Batch,M,1, R3] to [Batch,1,M, R3] to [Batch, M,M, R3] 
    vec_diff = input_xyz[...,None,:]-input_xyz[...,None,:,:]
    dist = np.sqrt(np.sum(np.square(vec_diff),axis=len(input_xyz.shape)))
    furthest_dist = np.max(dist)
    centroid  = get_midpoint(input_xyz)
    if print_dist:
        print(f'largest distance {furthest_dist:0.1f}')
    
    xyz_mean_zero = input_xyz - centroid[:,None,:]
    return xyz_mean_zero/furthest_dist

#?dgl.nn.pytorch.KNNGraph, nearest neighbor graph maker
def define_graph(batch_size=8,n_nodes=65):
    
    v1,v2,edge_data, ind = define_graph_edges(n_nodes)
    pe = make_pe_encoding(n_nodes=n_nodes)
    
    graphList = []
    
    for i in range(batch_size):
        
        g = dgl.graph((v1,v2))
        g.edata['con'] = edge_data
        g.ndata['pe'] = pe

        graphList.append(g)
        
    batched_graph = dgl.batch(graphList)

    return batched_graph
