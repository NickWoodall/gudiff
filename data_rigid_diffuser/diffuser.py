from data_rigid_diffuser import so3_diffuser
from data_rigid_diffuser import r3_diffuser
from scipy.spatial.transform import Rotation
from data_rigid_diffuser import rigid_utils as ru
import se3_diffuse.utils as du
import yaml
import torch
import numpy as np

def _extract_trans_rots(rigid: ru.Rigid):
    rot = rigid.get_rots().get_rot_mats().cpu().numpy()
    rot_shape = rot.shape
    num_rots = np.cumprod(rot_shape[:-2])[-1]
    rot = rot.reshape((num_rots, 3, 3))
    rot = Rotation.from_matrix(rot).as_rotvec().reshape(rot_shape[:-2] +(3,))
    tran = rigid.get_trans().cpu().numpy()
    return tran, rot

def _assemble_rigid(rotvec, trans):
    rotvec_shape = rotvec.shape
    num_rotvecs = np.cumprod(rotvec_shape[:-1])[-1]
    rotvec = rotvec.reshape((num_rotvecs, 3))
    rotmat = Rotation.from_rotvec(rotvec).as_matrix().reshape(
        rotvec_shape[:-1] + (3, 3))
    return ru.Rigid(
            rots=ru.Rotation(
                rot_mats=torch.Tensor(rotmat)),
            trans=torch.tensor(trans))

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
    """Generate Diffusion Noise based on FrameDiff"""
    
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
    
    def forward(self, bb_dict, t_vec=None, useR3=True , cast=torch.float32):
        
        ca = bb_dict['CA']
        nc_vec = bb_dict['N_CA'].reshape((-1,3))
        cc_vec = bb_dict['C_CA'].reshape((-1,3))
        
        if t_vec is None:
            t_vec =  np.random.uniform(size=ca.shape[0])
        score_scales = [self.score_scaling(t, useR3) for t in t_vec]
        
        #sample rotation
        rot_vec = np.array([self.so3d.sample(t, n_samples=ca.shape[1]) for t in t_vec]).reshape((-1,3))
        rotmat = Rotation.from_rotvec(rot_vec).as_matrix()
        
        batch_shape =  bb_dict['N_CA'].shape
        nc_vec_noised = ru.rot_vec_mul(torch.tensor(rotmat,dtype=cast),nc_vec).reshape(batch_shape)
        cc_vec_noised = ru.rot_vec_mul(torch.tensor(rotmat,dtype=cast),cc_vec).reshape(batch_shape)
        
        #get apply transation to CA
        ca_noised = np.array([self.r3d.forward_marginal(ca[i].numpy(),t)[0] for i,t in enumerate(t_vec)])
        ca_noised = torch.tensor(ca_noised, dtype=cast)
        
        bb_noised_out = {'CA': ca_noised, 'N_CA': nc_vec_noised, 'C_CA': cc_vec_noised}
        
        return bb_noised_out, torch.tensor(t_vec,dtype=cast), torch.tensor(score_scales,dtype=cast)
    
    def score(self, rigids_cur, rigids_init, batched_t):
        trans_score = self.r3d.score(rigids_cur.get_trans().cpu(), 
                                    rigids_init.get_trans().cpu(), 
                                    batched_t[:,None,None].repeat((1,rigids_cur.shape[1],3)).cpu())
        
        
        rots_t = rigids_init.get_rots()
        rots_0 = rigids_cur.get_rots()

        rots_0_inv = rots_0.invert()
        quats_0_inv = rots_0_inv.get_quats()
        quats_t = rots_t.get_quats()
        quats_0t = ru.quat_multiply(quats_0_inv, quats_t)
        rotvec_0t = du.quat_to_rotvec(quats_0t)
        rot_score = self.so3d.torch_score(rotvec_0t, batched_t)
        
        return trans_score, rot_score
    
    def forward_single_t(self, bb_dict, t, cast=torch.float32):
        
        ca = bb_dict['CA']
        nc_vec = bb_dict['N_CA'].reshape((-1,3))
        cc_vec = bb_dict['C_CA'].reshape((-1,3))
        #sample rotation
        n_samples = nc_vec.shape[0]
        rotvec = self.so3d.sample(t, n_samples=n_samples)
        rotmat = Rotation.from_rotvec(rotvec).as_matrix()
        #apply rotation
        
        batch_shape =  bb_dict['N_CA'].shape
        
        nc_vec_noised = ru.rot_vec_mul(torch.tensor(rotmat,dtype=cast),nc_vec).reshape(batch_shape)
        cc_vec_noised = ru.rot_vec_mul(torch.tensor(rotmat,dtype=cast),cc_vec).reshape(batch_shape)
        
        #r3d requires numpy
        ca_noised , _  = self.r3d.forward_marginal(ca.numpy(), t)
        ca_noised = torch.tensor(ca_noised, dtype=cast)
        
        bb_noised_out = {'CA': ca_noised, 'N_CA': nc_vec_noised, 'C_CA': cc_vec_noised}
        
        return bb_noised_out
    
 
    