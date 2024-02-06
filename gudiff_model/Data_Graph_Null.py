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
    
def circular_pe_encoding(n_nodes=128,embed_dim=12, cast_type=torch.float32):
    #positional encoding of node, all repeat every n_nodes
    i_array = np.arange(1,(embed_dim/2)+1)
    period = 2*np.pi
    wk = period/(n_nodes)*i_array**2
    t_array = np.arange(n_nodes)
    si = torch.tensor(np.sin(wk*t_array.reshape((-1,1))))
    ci = torch.tensor(np.cos(wk*t_array.reshape((-1,1))))
    pe = torch.stack((si,ci),axis=2).reshape(t_array.shape[0],embed_dim).type(cast_type)
    
    return pe

def monomer_null_knngraph(xyz,real_mask, k=2):
    """Takes in array of XYZ coordinates with null nodes repeated at termini.
    
    #Parameters
    real_mask: mask determining residues of real and null nodes
    
    Returns Graph that has a linear connections between all nodes indices and knn graph based on real nodes distance"""
    
    #make knn_graph from real nodes only
    #THERE ARE k PREDECESSORS FOR EACH NODE, SO CHECK DST 
    g_knn=dgl.knn_graph(xyz[real_mask],k=k,exclude_self=True) #this will make a batched 3d tensor if youwant

    #prepare adjacent node graph that connects real to null and (all nodes)
    n_nodes = xyz.shape[0]


    #create arrays to shift node indices_ from knn_graph based at zero back to original number
    knn_shift = torch.zeros((n_nodes,))

    #use mask to assign nodes as real or null
    #pulls the nodes out, as the real nodes will be assigned 0-len(real_nodes), and len(real_nodes) to end
    #like what happens with g_knn graph creation, where index of real_nodes_src convert to node numbering in graph
    #then null_nodes_src indexing + (len(real_nodes)) is equal to null node numbering in graph
    node_index =  torch.arange(n_nodes)
    real_nodes = node_index[real_mask==True]
    null_nodes = node_index[real_mask==False]

    # add null nodes to the end
    g_knn.add_nodes(len(null_nodes))
    #assign shifts as described above, from creating the graph will real nodes first then adding null nodes
    #the real nodes will be assigned 0-len(real_nodes), and len(real_nodes) to end
    knn_shift[:len(real_nodes)] = real_nodes
    knn_shift[len(real_nodes):] = null_nodes

    #convert the edges of knn_real+graph with adjacent graph added on top using knn_shift
    #to a new src, dst graph (src_conv,dst_conv) that refers to the original node numbering
    g_knn_src, g_knn_dst = g_knn.edges()

    conv_src = torch.ones_like(g_knn_src,dtype=torch.int)
    conv_dst = torch.ones_like(g_knn_dst,dtype=torch.int)

    #index i of src_knn_shift = (new knn_graph node numbering, value at )
    #(old graph node number) =  src_knn_shift[i] 
    #the real nodes will be assigned 0-len(real_nodes), and len(real_nodes) to end for null nodes
    #set the real nodes first, then the null nodes
    for i in range(real_nodes.shape[0]):
        conv_src[g_knn_src==i] = knn_shift[i]
        conv_dst[g_knn_dst==i] = knn_shift[i]

    for j in range(1,null_nodes.shape[0]+1):
        conv_src[g_knn_src==(i+j)] = knn_shift[i+j]
        conv_dst[g_knn_dst==(i+j)] = knn_shift[i+j]



    g = dgl.graph((conv_src,conv_dst))

    src_start = torch.arange(n_nodes,dtype=torch.int) #positive direction
    dst_start = torch.roll(src_start,-1) #negative dir
    #add adjacent graph on top, simple remove repeats
    g.add_edges(src_start,dst_start)
    g_out = dgl.to_simple(g)
    return g_out

def decirc(graph_in):
    """Remove circular connections. Last node to first node"""
    src,dst = graph_in.edges()

    circfw = (src==graph_in.nodes().max()) & (dst==0) 
    circbw = (src==0) & (dst==graph_in.nodes().max())
    both_circ = circfw | circbw
    
    src_out = src[~both_circ]
    dst_out = dst[~both_circ]
    
    return dgl.graph((src_out,dst_out))

def concat_monomer_graphs(graph_list):
    """Add single connection between each graph"""
    
    batched_graph = dgl.batch([decirc(g) for g in graph_list])
    src = torch.zeros((len(graph_list),),dtype=torch.int64) 
    dst = torch.zeros_like(src)
    
    #recirc concats
    nindex = 0
    src[nindex] = 0
    dst[nindex] = len(batched_graph.nodes())-1 #
    
    
    for i in range(1,len(graph_list)):
        src[i] = graph_list[nindex].nodes().max()+nindex
        dst[i] = src[i]+1
        nindex = src[i]
    
    
    batched_graph.add_edges(src,dst)
    batched_graph.add_edges(dst,src)
    
    return batched_graph

    
    
#this should probably be already transferred to GPU for speed... at some point
#this should probably be already transferred to GPU for speed... at some point
class Make_nullKNN_MP_Graphs():
    
    #8 long positional encoding
    NODE_FEATURE_DIM_0 = 17 #circular pe encoding dim
    EDGE_FEATURE_DIM = 1 # 0 or 1 primary seq connection or not
    NODE_FEATURE_DIM_1 = 2
    
    def __init__(self, KNN=30, mp_stride=4, n_nodes=128, coord_div=10, 
                       cast_type=torch.float32, channels_start=32,
                       ndf1=6, ndf0=32,embed_dim_pe=12, nr_node_feats=5,cuda=True):
        
        self.KNN = KNN
        self.n_nodes = n_nodes
        self.pe = circular_pe_encoding(n_nodes=n_nodes,embed_dim=embed_dim_pe, cast_type=torch.float32)
        self.mp_stride = mp_stride
        self.null_stride = mp_stride*2
        self.cast_type = cast_type
        self.channels_start = channels_start
        
        
        
        self.cuda = cuda
        self.ndf1 = ndf1 #awkard adding of nodes features to mpGraph
        self.ndf0 = ndf0
        self.NODE_FEATURE_DIM_0 = embed_dim_pe + nr_node_feats
        
        
    def create_and_batch(self, bb_dict, print_out=False,cast_type=torch.float):

        graphList = []
        mpGraphList = []
        mpRevGraphList = []
        mpSelfGraphList = []
        
        for j, caXYZ in enumerate(bb_dict['bb_noised']['CA']):
            #round nodes to be real (1) or (null 0)
            #get null node indices from mask
            real_nodes_feats = torch.round(bb_dict['real_nodes_noise'][j]).clamp(0,1)
            real_nodes_mask = real_nodes_feats.sum(-1)>1.99

            #make a knn graph form the real nodes only
            graph = monomer_null_knngraph(caXYZ, real_nodes_mask, k=self.KNN)
            graph.ndata['pe_nf'] = torch.cat((self.pe,real_nodes_feats),dim=-1).type(cast_type)
            graph.ndata['pos'] = caXYZ
            graph.ndata['bb_ori'] = torch.cat((bb_dict['bb_noised']['N_CA'][j], bb_dict['bb_noised']['C_CA'][j]),axis=1)
            graph.ndata['real_nodes_mask']=real_nodes_mask

            #gather edge data from all possible noised edges produced
            gsrc, gdst = graph.edges()
            num_edges = len(gsrc)

            #adjacent AA are one apart, or the loop connection from zero node to the last node
            adj_nodes_mask = ((torch.abs(gsrc-gdst)==1) | (torch.abs(gsrc-gdst)==len(gsrc)-1)) 

            #actually we need to determine
            null_nodes = torch.arange(self.n_nodes,dtype=torch.int)[~real_nodes_mask]
            real_nodes = torch.arange(self.n_nodes,dtype=torch.int)[real_nodes_mask]
            
                        #broadcast each src/dst node against all null nodes, if any match along node dimension is a null edge
            gsrc_compare = (gsrc[:,None] - null_nodes[None,:])
            gdst_compare = (gdst[:,None] - null_nodes[None,:])
            null_edges_src_ind = torch.where(gsrc_compare==0,True,False).any(dim=1)
            null_edges_dst_ind = torch.where(gdst_compare==0,True,False).any(dim=1)

            #broadcast each src/dst node against all real or null nodes, if any match along node dimension is a null edge
            gsrc_compare = (gsrc[:,None] - real_nodes[None,:])
            gdst_compare = (gdst[:,None] - real_nodes[None,:])
            real_edges_src_ind = torch.where(gsrc_compare==0,True,False).any(dim=1)
            real_edges_dst_ind = torch.where(gdst_compare==0,True,False).any(dim=1)
            #null edges connect to any null nodes (|), real edges must both be real (&)
            null_edges = null_edges_src_ind | null_edges_dst_ind
            real_edges = real_edges_src_ind & real_edges_dst_ind
            adj_real = adj_nodes_mask&real_edges

            #edge features likely not needed, taken care of in node features

            EDGE_DIM=self.EDGE_FEATURE_DIM

            ###
#             enoised = noised_dict['edge_cons'][j]
#             enoised = enoised.reshape((-1,EDGE_DIM,2))[:num_edges]
            edata = torch.ones(gsrc.shape+(EDGE_DIM,))
            # oh1_mask = torch.tensor([1,0,0]).unsqueeze(-1)
            # oh2_mask = torch.tensor([0,1,0]).unsqueeze(-1)
            # oh3_mask = torch.tensor([0,0,1]).unsqueeze(-1)


            # #needs to add - /+ N/C directon for this data????, or is this covered by pe encoding
            # edata[real_edges] = torch.gather(enoised[real_edges],2,oh2_mask.repeat(real_edges.sum(),1,1))[:,:,0]
            # edata[adj_real] = torch.gather(enoised[adj_real],2,oh1_mask.repeat(adj_real.sum(),1,1))[:,:,0]
            # edata[null_edges] = torch.gather(enoised[null_edges],2,oh3_mask.repeat(null_edges.sum(),1,1))[:,:,0]

            # edge_dir = torch.ones_like(gsrc,dtype=torch.long) #NotC direction off AA sequence
            # edge_dir[gdst<gsrc]=-1 # #NotC; reverse direction 
            # edata[adj_nodes_mask]=edata[adj_nodes_mask]*edge_dir

            graph.edata['con'] = edata

            #way to get real_edges and null edges in loop code
            # null_edges = torch.zeros_like(gsrc)
            # for i,x in enumerate(gsrc):
            #     if gsrc[i] in null_nodes or gdst[i] in null_nodes:
            #         null_edges[i] = 1

            # real_edges = torch.zeros_like(gsrc)
            # for i,x in enumerate(gsrc):
            #     if gsrc[i] in real_nodes and gdst[i] in real_nodes:
            #         real_edges[i] = 1

            #max possible mp feats, is +1 for range(start,end,stride) combines with real+null=total + stride rounding 
            mp_list = torch.zeros((len(list(range(0,self.n_nodes, self.mp_stride)))+1),caXYZ.shape[1])

            new_src = torch.tensor([],dtype=torch.int)
            new_dst = torch.tensor([],dtype=torch.int)

            new_src_rev = torch.tensor([], dtype=torch.int)
            new_dst_rev = torch.tensor([], dtype=torch.int)

            #create midpoints for real nodes
            i=0#mp list counter
            mp_real_node_counter = 0
            for real_index in range(0,len(real_nodes), self.mp_stride):
                x = real_nodes[real_index] #convert to match torch.int from
                src, dst = graph.in_edges(x) #dst repeats x, this grab null nodes too

                n_tot = torch.cat((x.unsqueeze(0),src)) #add x to node list
                mp_list[i] = caXYZ[n_tot].sum(axis=0)/n_tot.shape[0]
                mp_node = i + graph.num_nodes() #add midpoints nodes at end of graph
                mp_real_node_counter += 1
                #define edges between midpoint nodes and nodes defining midpoint for midpointGraph

                new_src = torch.cat((new_src,n_tot))
                new_dst = torch.cat((new_dst,
                                     (torch.tensor(mp_node,dtype=torch.int).unsqueeze(0).repeat(n_tot.shape[0]))))

                i+=1

            #remove extra null nodes from .in_edges call

            #remove edges that are null node connections,
            #dst are the midpoint nodes for mpGraph, src are mp nodes for mpGraphRev
            #only remove non-mp nodes

            real_mask_rem1 = torch.isin(new_src,real_nodes)
            #             real_mask_rem2 = torch.isin(new_dst_rev,real_nodes) 
            new_src = new_src[real_mask_rem1]
            new_dst = new_dst[real_mask_rem1]
            #             new_src_rev = new_src_rev[real_mask_rem2]
            #             new_dst_rev = new_dst_rev[real_mask_rem2]


            #collapse collected null nodes onto null mp of contingous sections 
            end_p = ((null_nodes.roll(1)-null_nodes)==-1) #consecutive are equal to negative one (look right)
            start_p = ((null_nodes.roll(-1)-null_nodes)==1) #consecutive are equal to one (look left)
            startend = (start_p != end_p) #remove overlap of interior consecutive nodes
            start = start_p == startend #just get the starts
            end  = end_p == startend #just get the ends
            si = torch.arange(len(start),dtype=torch.int)[start]# indices of start of consecutive nodes
            ei = torch.arange((len(end)),dtype=torch.int)[end] # indices of end of cone

            #connect first and last groups if approriate
            if null_nodes[0]==0 and null_nodes[-1]==self.n_nodes-1:
                #roll last group across barrier
                roll_con = len(start)-si[-1]
                null_nodes = null_nodes.roll(int(roll_con))
                #update end index and start index by roll and remove groups (one from end)
                #add zero to start and remove last start (rolled)
                ei = (ei+roll_con)[:-1]
                sic=torch.zeros_like(si[1:])
                sic[1:] = si[1:-1]+roll_con
                si = sic

            #mp_list_null  = torch.ones((si.shape[0],caXYZ.shape[1]))*-1e9
            #add null nodes to the end of mp_list
            counter_mp_index = 0 #mp list counter, start/end  
            tot_indices = si.shape[0]
            while counter_mp_index < tot_indices:

                n_tot = null_nodes[si[counter_mp_index]:ei[counter_mp_index]+1]
                while len(n_tot) <  self.mp_stride and counter_mp_index+1<tot_indices:
                    #merge non-continuous null nodes smaller than stride
                    counter_mp_index=counter_mp_index+1
                    n_tot = torch.cat([n_tot,null_nodes[si[counter_mp_index]:ei[counter_mp_index]+1]],axis=0)

                mp_list[i] = caXYZ[n_tot].sum(axis=0)/n_tot.shape[0]
                mp_node = i + graph.num_nodes() #add midpoints nodes at end of graph

                #from null nodes to new mp_node
                new_src = torch.cat((new_src,n_tot))
                new_dst = torch.cat((new_dst,
                                     (torch.tensor(mp_node,dtype=torch.int).unsqueeze(0).repeat(n_tot.shape[0]))))
                #and reverse graph for coming off
                new_src_rev = torch.cat((new_src_rev,
                                         (torch.tensor(mp_node,dtype=torch.int).unsqueeze(0).repeat(n_tot.shape[0]))))
                new_dst_rev = torch.cat((new_dst_rev,n_tot))

                i=i+1
                counter_mp_index += 1

            mp_node_indx = torch.arange(0,len(mp_list)).type(torch.int)    

            mpGraph = dgl.graph((new_src,new_dst))
            to_Add = len(mp_list)+graph.num_nodes()-mpGraph.num_nodes()
            mpGraph.add_nodes(to_Add) #nodes without any use for padding
            mp_pos = torch.cat((caXYZ,mp_list),axis=0).type(self.cast_type)
            mpGraph.ndata['pos'] = mp_pos

            mp_real_node_counter
            mp_real_node_counter = counter_mp_index

            #mp real/ null nodes
            mp_node_real_mask = torch.zeros(mp_list.shape[0],dtype=torch.bool)
            mp_node_real_mask[:mp_real_node_counter] = True
            mpGraph.ndata['mp_node_real_mask'] = torch.cat([real_nodes_mask,mp_node_real_mask])

            #match output shape of first transformer
            pe_mp = torch.cat((self.pe,torch.zeros((self.pe.shape[0], self.channels_start-self.pe.shape[1]))),axis=1)
            mpGraph.ndata['pe'] = torch.cat((pe_mp,pe_mp[mp_node_indx]))
            mpGraph.edata['con'] = torch.ones((mpGraph.num_edges(),1))
            mpGraph_rev = dgl.graph((new_dst,new_src))
            mpGraph_rev.add_nodes(to_Add)
            mpGraph_rev.ndata['pos'] = torch.cat((caXYZ,mp_list),axis=0).type(self.cast_type)
            mpGraph_rev.ndata['pe'] = torch.cat((pe_mp,pe_mp[mp_node_indx]))
            mpGraph_rev.edata['con'] = torch.ones((mpGraph_rev.num_edges(),1))
            mpGraph_rev.ndata['mp_node_real_mask'] = torch.cat([real_nodes_mask,mp_node_real_mask])
            #make graph for self interaction of midpoints
            v1,v2,edge_data, ind = define_graph_edges(len(mp_list))
            mpSelfGraph = dgl.graph((v1,v2))
            mpSelfGraph.edata['con'] = edge_data.reshape((-1,1))
            mpSelfGraph.ndata['pe'] = self.pe[mp_node_indx] #not really needed
            mpSelfGraph.ndata['pos'] = mp_list.type(self.cast_type)

            
            mpSelfGraphList.append(mpSelfGraph)
            mpGraphList.append(mpGraph)
            mpRevGraphList.append(mpGraph_rev)
            graphList.append(graph)
       
        return dgl.batch(graphList), dgl.batch(mpGraphList), dgl.batch(mpSelfGraphList), dgl.batch(mpRevGraphList)
    
    def prep_for_network(self, bb_dict, cuda=True):
    
        batched_graph, batched_mpgraph, batched_mpself_graph, batched_mpRevgraph =  self.create_and_batch(bb_dict)
        
        edge_feats        =    {'0':   batched_graph.edata['con'][:, :self.EDGE_FEATURE_DIM, None]}
        edge_feats_mp     = {'0': batched_mpgraph.edata['con'][:, :self.EDGE_FEATURE_DIM, None]} #def all one now
        edge_feats_mpself = {'0': batched_mpself_graph.edata['con'][:, :self.EDGE_FEATURE_DIM, None]}
#         edge_feats_mp     = {'0': batched_mpRevgraph.edata['con'][:, :self.EDGE_FEATURE_DIM, None]}
        batched_graph.edata['rel_pos']   = _get_relative_pos(batched_graph)
        batched_mpgraph.edata['rel_pos'] = _get_relative_pos(batched_mpgraph)
        batched_mpself_graph.edata['rel_pos'] = _get_relative_pos(batched_mpself_graph)
        batched_mpRevgraph.edata['rel_pos'] = _get_relative_pos(batched_mpRevgraph)
        # get node features
        node_feats =         {'0': batched_graph.ndata['pe_nf'][:, :self.NODE_FEATURE_DIM_0, None],
                              '1': batched_graph.ndata['bb_ori'][:,:self.NODE_FEATURE_DIM_1, :3]}
        node_feats_mp =      {'0': batched_mpgraph.ndata['pe'][:, :self.ndf0, None],
                              '1': torch.ones((batched_mpgraph.num_nodes(),self.ndf1,3))}
        #unused
        node_feats_mpself =  {'0': batched_mpself_graph.ndata['pe'][:, :self.NODE_FEATURE_DIM_0, None]}
        
        out_dict = {}
        
        if cuda:
            bg,nf,ef = to_cuda(batched_graph), to_cuda(node_feats), to_cuda(edge_feats)
            bg_mp, nf_mp, ef_mp = to_cuda(batched_mpgraph), to_cuda(node_feats_mp), to_cuda(edge_feats_mp)
            bg_mps, nf_mps, ef_mps = to_cuda(batched_mpself_graph), to_cuda(node_feats_mpself), to_cuda(edge_feats_mpself)
            bg_mpRev = to_cuda(batched_mpRevgraph)

            
            #return bg,nf,ef, bg_mp, nf_mp, ef_mp, bg_mps, nf_mps, ef_mps, bg_mpRev
        
        else:
            bg,nf,ef = batched_graph, node_feats, edge_feats
            bg_mp, nf_mp, ef_mp = batched_mpgraph, node_feats_mp, edge_feats_mp
            bg_mps, nf_mps, ef_mps = batched_mpself_graph, node_feats_mpself, edge_feats_mpself
            bg_mpRev = batched_mpRevgraph
            
            #return bg,nf,ef, bg_mp, nf_mp, ef_mp, bg_mps, nf_mps, ef_mps, bg_mpRev
        
                    
        out_dict['batched_graph'] = bg
        out_dict['node_feats'] = nf
        out_dict['edge_feats'] = ef
        out_dict['batched_graph_mp'] = bg_mp
        out_dict['node_feats_mp'] = nf_mp
        out_dict['edge_feats_mp'] = ef_mp
        out_dict['batched_graph_mpself'] = bg_mps
        out_dict['node_feats_mpself'] = nf_mps
        out_dict['edge_feats_mpself'] = ef_mps
        out_dict['batched_graph_mprev'] = bg_mpRev
        
        return out_dict

        
            

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
        