from data_rigid_diffuser import so3_diffuser
from data_rigid_diffuser import r3_diffuser
from scipy.spatial.transform import Rotation
from data_rigid_diffuser import rigid_utils as ru
import yaml
import torch
import numpy as np

# Applies to Python-3 Standard Library
class Struct(object):
    def __init__(self, data):
        for name, value in data.items():
            setattr(self, name, self._wrap(value))

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)): 
            return type(value)([self._wrap(v) for v in value])
        else:
            return Struct(value) if isinstance(value, dict) else value

class FrameDiffNoise(torch.nn.Module):
    """Generate Diffusion Noise based on FrameDiff.
    
    ca_nodes: number of calpha nodes
    camp_nodes: number of calpha nodes+mp nodes (getting the size of the t vector)
    
    
    """
    
    def __init__(self, config_path='data_rigid_diffuser/base.yaml'):
        super().__init__()
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        conf = Struct(config['diffuser'])
        
        self.so3d = so3_diffuser.SO3Diffuser(conf.so3)
        self.r3d =  r3_diffuser.R3Diffuser(conf.r3)
        
    def score_scaling(self, t, useR3 = True):
        if useR3:
            score_scaling = self.r3d.score_scaling(t)
        else:
            score_scaling = self.so3d.score_scaling(t)
        return score_scaling
    
    def noise_visualize(self, bb_dict, t_vec=None):
        """"Produce various t-samptes to check model. Length a multiple of the batch size"""
        
        cast = torch.float
        t_vec = np.array([0.01,0.05,0.01,0.2, 0.3,0.5,0.75,1])

        batch_size = len(t_vec)

        ca =     bb_dict['CA'][0][None,...].repeat(len(t_vec),1,1)
        nc_vec = bb_dict['N_CA'][0][None,...].repeat(len(t_vec),1,1,1).reshape((-1,3))
        cc_vec = bb_dict['C_CA'][0][None,...].repeat(len(t_vec),1,1,1).reshape((-1,3))

        out_shape = bb_dict['N_CA'][0][None,...].repeat(len(t_vec),1,1,1).shape

        #sample rotation
        rot_vec = np.array([self.so3d.sample(t) for t in t_vec]).squeeze(1)
        rotmat = Rotation.from_rotvec(rot_vec).as_matrix()
        #apply rotation
        rotmat = rotmat[:,None,...].repeat(ca.shape[1],axis=1).reshape((-1,3,3))
        nc_vec_noised = ru.rot_vec_mul(torch.tensor(rotmat,dtype=cast),nc_vec).reshape(out_shape)
        cc_vec_noised = ru.rot_vec_mul(torch.tensor(rotmat,dtype=cast),cc_vec).reshape(out_shape)

        #get apply transation to CA
        ca_noised = np.array([self.r3d.forward_marginal(ca[i].numpy(),t)[0] for i,t in enumerate(t_vec)])
        ca_noised = torch.tensor(ca_noised, dtype=cast)

        bb_noised_out = {'CA': ca_noised, 'N_CA': nc_vec_noised, 'C_CA': cc_vec_noised}
        
        return bb_noised_out, torch.tensor(t_vec,dtype=cast)
        
    
    def forward(self, bb_dict, t_vec = None, t_mult=1, cast=torch.float32):
        
        ca = bb_dict['CA']
        nc_vec = bb_dict['N_CA'].reshape((-1,3))
        cc_vec = bb_dict['C_CA'].reshape((-1,3))
        
        out_shape = bb_dict['N_CA'].shape
        
        batch_size = ca.shape[0]
        if t_vec == None:
            t_vec =  np.random.uniform(size=batch_size)*t_mult
        score_scales = [self.score_scaling(t) for t in t_vec]
        
        #sample rotation
        rot_vec = np.array([self.so3d.sample(t) for t in t_vec]).squeeze(1)
        rotmat = Rotation.from_rotvec(rot_vec).as_matrix()
        #apply rotation
        rotmat = rotmat[:,None,...].repeat(ca.shape[1],axis=1).reshape((-1,3,3))
        nc_vec_noised = ru.rot_vec_mul(torch.tensor(rotmat,dtype=cast),nc_vec).reshape(out_shape)
        cc_vec_noised = ru.rot_vec_mul(torch.tensor(rotmat,dtype=cast),cc_vec).reshape(out_shape)
        
        #get apply transation to CA
        ca_noised = np.array([self.r3d.forward_marginal(ca[i].numpy(),t)[0] for i,t in enumerate(t_vec)])
        ca_noised = torch.tensor(ca_noised, dtype=cast)
        
        bb_noised_out = {'CA': ca_noised, 'N_CA': nc_vec_noised, 'C_CA': cc_vec_noised}
        
        return bb_noised_out, torch.tensor(t_vec,dtype=cast), torch.tensor(score_scales,dtype=cast)