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
from se3_transformer.runtime.utils import to_cuda
from se3_diffuse import rigid_utils as ru



# N_CA_dist = torch.tensor(1.458/10.0).to('cuda')
# C_CA_dist = torch.tensor(1.523/10.0).to('cuda')
N_CA_dist = torch.tensor(1.458) #update this when imported
C_CA_dist = torch.tensor(1.523)

#Globals from npose

if ( hasattr(os, 'ATOM_NAMES') ):
    assert( hasattr(os, 'PDB_ORDER') )

    ATOM_NAMES = os.ATOM_NAMES
    PDB_ORDER = os.PDB_ORDER
else:
    ATOM_NAMES=['N', 'CA', 'CB', 'C', 'O']
    PDB_ORDER = ['N', 'CA', 'C', 'O', 'CB']

_byte_atom_names = []
_atom_names = []
for i, atom_name in enumerate(ATOM_NAMES):
    long_name = " " + atom_name + "       "
    _atom_names.append(long_name[:4])
    _byte_atom_names.append(atom_name.encode())

    globals()[atom_name] = i

R = len(ATOM_NAMES)

if ( "N" not in globals() ):
    N = -1
if ( "C" not in globals() ):
    C = -1
if ( "CB" not in globals() ):
    CB = -1


_pdb_order = []
for name in PDB_ORDER:
    _pdb_order.append( ATOM_NAMES.index(name) )

#-------------------------Helper functions 
def define_graph_edges(n_nodes):
    #input number of nodes: n_nodes
    #connected backbone = 1, unconnected = 0

    con_v1 = np.arange(n_nodes-1) #vertex 1 of edges in chronological order
    con_v2 = np.arange(1,n_nodes) #vertex 2 of edges in chronological order

    ind = con_v1*(n_nodes-1)+con_v2-1 #account for removed self connections (-1)


    #unconnected backbone

    nodes = np.arange(n_nodes)
    v1 = np.repeat(nodes,n_nodes-1) #starting vertices, same number repeated for each edge

    start_v2 = np.repeat(np.arange(n_nodes)[None,:],n_nodes,axis=0)
    diag_ind = np.diag_indices(n_nodes)
    start_v2[diag_ind] = -1 #diagonal of matrix is self connections which we remove (self connections are managed by SE3 Conv channels)
    v2 = start_v2[start_v2>-0.5] #remove diagonal and flatten

    edge_data = torch.zeros(len(v2))
    edge_data[ind] = 1
    
    return v1,v2,edge_data, ind

def make_pe_encoding(n_nodes=65, embed_dim = 12, scale = 40, cast_type=torch.float32, print_out=False):
    #positional encoding of node, scale dimension optimized for 65
    i_array = np.arange(1,(embed_dim/2)+1)
    wk = (1/(scale**(i_array*2/embed_dim)))
    t_array = np.arange(n_nodes)
    si = torch.tensor(np.sin(wk*t_array.reshape((-1,1))))
    ci = torch.tensor(np.cos(wk*t_array.reshape((-1,1))))
    pe = torch.stack((si,ci),axis=2).reshape(t_array.shape[0],embed_dim).type(cast_type)
    
    if print_out == True:
        for x in range(int(n_nodes/12)):
            print(np.round(pe[x],1))
    
    return pe

def _get_relative_pos(graph_in: dgl.DGLGraph) -> torch.Tensor:
    x = graph_in.ndata['pos']
    src, dst = graph_in.edges()
    rel_pos = x[dst] - x[src]
    return rel_pos

def torch_normalize(v, eps=1e-6):
    """Normalize vector in last axis"""
    norm = torch.linalg.vector_norm(v, dim=len(v.shape)-1)+eps
    return v / norm[...,None]

def normalize(v):
    """Normalize vector in last axis"""
    norm = np.linalg.norm(v,axis=len(v.shape)-1)
    norm[norm == 0] = 1
    return v / norm[...,None]

def pad_n_rows_below_2d_matrix(arr, n, pad_num=-1e9):
    """Adds n rows of zeros below 2D numpy array matrix.

    :param arr: A two dimensional numpy array that is padded.
    :param n: the number of rows that are added below the matrix.
    """
    padded_array = np.ones((arr.shape[0] + n,)+ arr.shape[1:])*pad_num
    padded_array[: arr.shape[0], :] = arr
    return padded_array

class ProteinBB_Dataset(Dataset):
    def __init__(self, coordinates_list: [np.array], n_nodes=128,
                 n_atoms=5, coord_div=10, center_mass=True,  cast_type=torch.float32):
        #prots,#length_prot in aa, #residues/aa, #xyz per atom
           
        #alphaFold reduce by 10
        self.coord_div = coord_div
        self.n_atoms = 5 #see npose utils
        
        
        
        self.lenlist = [len(x) for x in coordinates_list]
        
        self.prot_coords = np.zeros((len(coordinates_list),n_nodes,n_atoms,3))
        
        for i,coordinates in enumerate(coordinates_list):
            
            if center_mass:
                com  = (coordinates[:,CA,:].sum(axis=0)/coordinates[:,CA,:].shape[0])
                coordinates = coordinates-com
            
            coordinates = coordinates/coord_div
            self.prot_coords[i] = pad_n_rows_below_2d_matrix(coordinates, n_nodes-len(coordinates), pad_num=-1e9)

        
        self.ca_coords = torch.tensor(self.prot_coords[:,:,CA,:], dtype=cast_type)
        self.N_CA_vec = torch.tensor(self.prot_coords[:,:,N,:] - self.prot_coords[:,:,CA,:], dtype=cast_type)
        self.C_CA_vec = torch.tensor(self.prot_coords[:,:,C,:] - self.prot_coords[:,:,CA,:], dtype=cast_type)
        
        #unsqueeze to stack together later
        self.N_CA_vec = torch_normalize(self.N_CA_vec).unsqueeze(2)
        self.C_CA_vec = torch_normalize(self.C_CA_vec).unsqueeze(2)
        
    def __len__(self):
        return len(self.ca_coords)

    def __getitem__(self, idx):
        return {'CA':self.ca_coords[idx], 'N_CA':self.N_CA_vec[idx], 'C_CA':self.C_CA_vec[idx]}

class Helix4_Dataset(Dataset):
    def __init__(self, coordinates: np.array, cast_type=torch.float32):
        #prots,#length_prot in aa, #residues/aa, #xyz per atom
           
        #alphaFold reduce by 10
        coord_div = 10
        
        coordinates = coordinates/coord_div
        self.ca_coords = torch.tensor(coordinates[:,:,CA,:], dtype=cast_type)
        #unsqueeze to stack together later
        self.N_CA_vec = torch.tensor(coordinates[:,:,N,:] - coordinates[:,:,CA,:], dtype=cast_type)
        self.C_CA_vec = torch.tensor(coordinates[:,:,C,:] - coordinates[:,:,CA,:], dtype=cast_type)
        
        self.N_CA_vec = torch_normalize(self.N_CA_vec).unsqueeze(2)
        self.C_CA_vec = torch_normalize(self.C_CA_vec).unsqueeze(2)
        
    def __len__(self):
        return len(self.ca_coords)

    def __getitem__(self, idx):
        return {'CA':self.ca_coords[idx], 'N_CA':self.N_CA_vec[idx], 'C_CA':self.C_CA_vec[idx]}
    
class Helix4_Dataset_Score(Dataset):
    def __init__(self, coordinates: np.array, cast_type=torch.float32):
        #prots,#length_prot in aa, #residues/aa, #xyz per atom
           
        #alphaFold style reduce by 10
        coord_div = 10
        
        #center at zero
        com  = (coordinates[:,:,CA,:].sum(axis=1)[:,None,:]/coordinates[:,:,CA,:].shape[1])[:,None,:]
        coordinates = coordinates-com
        
        coordinates = coordinates/coord_div
        self.ca_coords = torch.tensor(coordinates[:,:,CA,:], dtype=cast_type)
        #unsqueeze to stack together later
        self.N_CA_vec = torch.tensor(coordinates[:,:,N,:] - coordinates[:,:,CA,:], dtype=cast_type)
        self.C_CA_vec = torch.tensor(coordinates[:,:,C,:] - coordinates[:,:,CA,:], dtype=cast_type)
        
        self.N_CA_vec = torch_normalize(self.N_CA_vec).unsqueeze(2)
        self.C_CA_vec = torch_normalize(self.C_CA_vec).unsqueeze(2)
        
        self.rigids = torch.concatenate((torch.tensor(coordinates[:,:,N,:], dtype=cast_type).unsqueeze(2),
                                         torch.tensor(coordinates[:,:,CA,:], dtype=cast_type).unsqueeze(2),
                                         torch.tensor(coordinates[:,:,C,:], dtype=cast_type).unsqueeze(2)), dim=-2)
        
        
    def __len__(self):
        return len(self.ca_coords)

    def __getitem__(self, idx):
        return {'CA':self.ca_coords[idx], 'N_CA':self.N_CA_vec[idx], 'C_CA':self.C_CA_vec[idx], 'rigids':self.rigids[idx]}
    
    

    
    
class Make_KNN_MP_Graphs():
    
    #8 long positional encoding
    NODE_FEATURE_DIM_0 = 12
    EDGE_FEATURE_DIM = 1 # 0 or 1 primary seq connection or not
    NODE_FEATURE_DIM_1 = 2
    
    def __init__(self, mp_stride=4, n_nodes=65, radius=15, coord_div=10, cast_type=torch.float32, channels_start=32,
                       ndf1=6, ndf0=32,cuda=True):
        
        self.KNN = 30
        self.n_nodes = n_nodes
        self.pe = make_pe_encoding(n_nodes=n_nodes)
        self.mp_stride = mp_stride
        self.cast_type = cast_type
        self.channels_start = channels_start
        
        self.cuda = cuda
        self.ndf1 = ndf1 #awkard adding of nodes features to mpGraph
        self.ndf0 = ndf0
        
    def create_and_batch(self, bb_dict):
        
        graphList = []
        mpGraphList = []
        mpRevGraphList = []
        mpSelfGraphList = []
        
        for j, caXYZ in enumerate(bb_dict['CA']):
            graph = dgl.knn_graph(caXYZ, self.KNN)
            graph.ndata['pe'] = self.pe.to(caXYZ.device)
            graph.ndata['pos'] = caXYZ
            graph.ndata['bb_ori'] = torch.cat((bb_dict['N_CA'][j],  bb_dict['C_CA'][j]),axis=1)
            
            #define covalent connections
            esrc, edst = graph.edges()
            graph.edata['con'] = (torch.abs(esrc-edst)==1).type(self.cast_type).reshape((-1,1))
            
            mp_list = torch.zeros((len(list(range(0,self.n_nodes, self.mp_stride))),caXYZ.shape[1]),device=caXYZ.device)
            
            new_src = torch.tensor([],dtype=torch.int,device=caXYZ.device)
            new_dst = torch.tensor([],dtype=torch.int,device=caXYZ.device)
            
            new_src_rev = torch.tensor([], dtype=torch.int,device=caXYZ.device)
            new_dst_rev = torch.tensor([], dtype=torch.int,device=caXYZ.device)
           
            i=0#mp list counter
            for x in range(0,self.n_nodes, self.mp_stride):
                src, dst = graph.in_edges(x) #dst repeats x
                n_tot = torch.cat((torch.tensor(x,device=caXYZ.device).unsqueeze(0),src)) #add x to node list
                mp_list[i] = caXYZ[n_tot].sum(axis=0)/n_tot.shape[0]
                mp_node = i + graph.num_nodes() #add midpoints nodes at end of graph
                #define edges between midpoint nodes and nodes defining midpoint for midpointGraph
                
                new_src = torch.cat((new_src,n_tot))
                new_dst = torch.cat((new_dst,
                                     (torch.tensor(mp_node,device=caXYZ.device).unsqueeze(0).repeat(n_tot.shape[0]))))
                #and reverse graph for coming off
                new_src_rev = torch.cat((new_src_rev,
                                         (torch.tensor(mp_node,device=caXYZ.device).unsqueeze(0).repeat(n_tot.shape[0]))))
                new_dst_rev = torch.cat((new_dst_rev,n_tot))
                
                i+=1
                
            mpGraph = dgl.graph((new_src,new_dst))
            mpGraph.ndata['pos'] = torch.cat((caXYZ,mp_list),axis=0).type(self.cast_type)
            mp_node_indx = torch.arange(0,self.n_nodes, self.mp_stride).type(torch.int)
            #match output shape of first transformer
            pe_mp = torch.cat(
                              (self.pe.to(caXYZ.device),
                              torch.zeros( (self.pe.shape[0],
                                            self.channels_start-self.pe.shape[1]),
                                                device=caXYZ.device))
                                                                     ,axis=1)
            mpGraph.ndata['pe'] = torch.cat((pe_mp,pe_mp[mp_node_indx]))
            mpGraph.edata['con'] = torch.zeros((mpGraph.num_edges(),1),device=caXYZ.device)
            
            mpGraph_rev = dgl.graph((new_src_rev,new_dst_rev))
            mpGraph_rev.ndata['pos'] = torch.cat((caXYZ,mp_list),axis=0).type(self.cast_type)
            mpGraph_rev.ndata['pe'] = torch.cat((pe_mp,pe_mp[mp_node_indx]))
            mpGraph_rev.edata['con'] = torch.zeros((mpGraph_rev.num_edges(),1),device=caXYZ.device)
            
            #make graph for self interaction of midpoints
            v1,v2,edge_data, ind = define_graph_edges(len(mp_list))
            mpSelfGraph = dgl.graph((v1,v2))
            mpSelfGraph.edata['con'] = edge_data.reshape((-1,1))
            mpSelfGraph.ndata['pe'] = self.pe[mp_node_indx] #not really needed
            mpSelfGraph = mpSelfGraph.to(caXYZ.device)
            mpSelfGraph.ndata['pos'] = mp_list.type(self.cast_type)
            
            
            mpSelfGraphList.append(mpSelfGraph) 
            mpGraphList.append(mpGraph)
            mpRevGraphList.append(mpGraph_rev)
            graphList.append(graph)
        
        return dgl.batch(graphList), dgl.batch(mpGraphList), dgl.batch(mpSelfGraphList), dgl.batch(mpRevGraphList)
    
    def prep_for_network(self, bb_dict, cuda=True):
    
        batched_graph, batched_mpgraph, batched_mpself_graph, batched_mpRevgraph =  self.create_and_batch(bb_dict)
        
        edge_feats        =    {'0':   batched_graph.edata['con'][:, :self.EDGE_FEATURE_DIM, None]}
        edge_feats_mp     = {'0': batched_mpgraph.edata['con'][:, :self.EDGE_FEATURE_DIM, None]} #def all zero now
        edge_feats_mpself = {'0': batched_mpself_graph.edata['con'][:, :self.EDGE_FEATURE_DIM, None]}
#         edge_feats_mp     = {'0': batched_mpRevgraph.edata['con'][:, :self.EDGE_FEATURE_DIM, None]}
        batched_graph.edata['rel_pos']   = _get_relative_pos(batched_graph)
        batched_mpgraph.edata['rel_pos'] = _get_relative_pos(batched_mpgraph)
        batched_mpself_graph.edata['rel_pos'] = _get_relative_pos(batched_mpself_graph)
        batched_mpRevgraph.edata['rel_pos'] = _get_relative_pos(batched_mpRevgraph)
        # get node features
        
        node_feats =         {'0': batched_graph.ndata['pe'][:, :self.NODE_FEATURE_DIM_0, None],
                              '1': batched_graph.ndata['bb_ori'][:,:self.NODE_FEATURE_DIM_1, :3]}
        node_feats_mp =      {'0': batched_mpgraph.ndata['pe'][:, :self.ndf0, None],
                              '1': torch.ones((batched_mpgraph.num_nodes(),self.ndf1,3))}
        #unused
        node_feats_mpself =  {'0': batched_mpself_graph.ndata['pe'][:, :self.NODE_FEATURE_DIM_0, None]}
        
        if cuda:
            bg,nf,ef = to_cuda(batched_graph), to_cuda(node_feats), to_cuda(edge_feats)
            bg_mp, nf_mp, ef_mp = to_cuda(batched_mpgraph), to_cuda(node_feats_mp), to_cuda(edge_feats_mp)
            bg_mps, nf_mps, ef_mps = to_cuda(batched_mpself_graph), to_cuda(node_feats_mpself), to_cuda(edge_feats_mpself)
            bg_mpRev = to_cuda(batched_mpRevgraph)
            
            return bg,nf,ef, bg_mp, nf_mp, ef_mp, bg_mps, nf_mps, ef_mps, bg_mpRev
        
        else:
            bg,nf,ef = batched_graph, node_feats, edge_feats
            bg_mp, nf_mp, ef_mp = batched_mpgraph, node_feats_mp, edge_feats_mp
            bg_mps, nf_mps, ef_mps = batched_mpself_graph, node_feats_mpself, edge_feats_mpself
            bg_mpRev = batched_mpRevgraph
            
            return bg,nf,ef, bg_mp, nf_mp, ef_mp, bg_mps, nf_mps, ef_mps, bg_mpRev
        
            

def get_edge_features(graph,edge_feature_dim=1):
    return {'0': graph.edata['con'][:, :edge_feature_dim, None]}

def define_poolGraph(n_nodes, batch_size, cast_type=torch.float32, cuda_out=True ):
    
    v1,v2,edge_data, ind = define_graph_edges(n_nodes)
    #pe = make_pe_encoding(n_nodes=n_nodes)#pe e
    
    graphList = []
    
    for i in range(batch_size):
        
        g = dgl.graph((v1,v2))
        g.edata['con'] = edge_data.type(cast_type).reshape((-1,1))
        g.ndata['pos'] = torch.zeros((n_nodes,3),dtype=torch.float32)

        graphList.append(g)
        
    batched_graph = dgl.batch(graphList)
    
    if cuda_out:
        return to_cuda(batched_graph)
    else:
        return batched_graph
    
def build_npose_from_coords(coords_in):
    """Use N, CA, C coordinates to generate O an CB atoms. """
    rot_mat_cat = np.ones(sum((coords_in.shape[:-1], (1,)), ()))
    
    coords = np.concatenate((coords_in,rot_mat_cat),axis=-1)
    
    npose = np.ones((coords_in.shape[0]*5,4)) #5 is atoms per res

    by_res = npose.reshape(-1, 5, 4)
    
    if ( "N" in ATOM_NAMES ):
        by_res[:,N,:3] = coords_in[:,0,:3]
    if ( "CA" in ATOM_NAMES ):
        by_res[:,CA,:3] = coords_in[:,1,:3]
    if ( "C" in ATOM_NAMES ):
        by_res[:,C,:3] = coords_in[:,2,:3]
    if ( "O" in ATOM_NAMES ):
        by_res[:,O,:3] = nu.build_O(npose)
    if ( "CB" in ATOM_NAMES ):
        tpose = nu.tpose_from_npose(npose)
        by_res[:,CB,:] = nu.build_CB(tpose)

    return npose

def dump_coord_pdb(coords_in, fileOut='fileOut.pdb'):
    
    npose =  build_npose_from_coords(coords_in)
    nu.dump_npdb(npose,fileOut)
        