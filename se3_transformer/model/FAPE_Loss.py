import torch
from se3_transformer.model.chemical import cos_ideal_NCAC #from RoseTTAFold2

#https://github.com/uw-ipd/RoseTTAFold2/blob/main/network/loss.py

def normQ(Q):
    """normalize a quaternions
    """
    return Q / torch.linalg.norm(Q, keepdim=True, dim=-1)

def Rs2Qs(Rs):
    Qs = torch.zeros((*Rs.shape[:-2],4), device=Rs.device)

    Qs[...,0] = 1.0 + Rs[...,0,0] + Rs[...,1,1] + Rs[...,2,2]
    Qs[...,1] = 1.0 + Rs[...,0,0] - Rs[...,1,1] - Rs[...,2,2]
    Qs[...,2] = 1.0 - Rs[...,0,0] + Rs[...,1,1] - Rs[...,2,2]
    Qs[...,3] = 1.0 - Rs[...,0,0] - Rs[...,1,1] + Rs[...,2,2]
    Qs[Qs<0.0] = 0.0
    Qs = torch.sqrt(Qs) / 2.0
    Qs[...,1] *= torch.sign( Rs[...,2,1] - Rs[...,1,2] )
    Qs[...,2] *= torch.sign( Rs[...,0,2] - Rs[...,2,0] )
    Qs[...,3] *= torch.sign( Rs[...,1,0] - Rs[...,0,1] )

    return Qs

def Qs2Rs(Qs):
    Rs = torch.zeros((*Qs.shape[:-1],3,3), device=Qs.device)

    Rs[...,0,0] = Qs[...,0]*Qs[...,0]+Qs[...,1]*Qs[...,1]-Qs[...,2]*Qs[...,2]-Qs[...,3]*Qs[...,3]
    Rs[...,0,1] = 2*Qs[...,1]*Qs[...,2] - 2*Qs[...,0]*Qs[...,3]
    Rs[...,0,2] = 2*Qs[...,1]*Qs[...,3] + 2*Qs[...,0]*Qs[...,2]
    Rs[...,1,0] = 2*Qs[...,1]*Qs[...,2] + 2*Qs[...,0]*Qs[...,3]
    Rs[...,1,1] = Qs[...,0]*Qs[...,0]-Qs[...,1]*Qs[...,1]+Qs[...,2]*Qs[...,2]-Qs[...,3]*Qs[...,3]
    Rs[...,1,2] = 2*Qs[...,2]*Qs[...,3] - 2*Qs[...,0]*Qs[...,1]
    Rs[...,2,0] = 2*Qs[...,1]*Qs[...,3] - 2*Qs[...,0]*Qs[...,2]
    Rs[...,2,1] = 2*Qs[...,2]*Qs[...,3] + 2*Qs[...,0]*Qs[...,1]
    Rs[...,2,2] = Qs[...,0]*Qs[...,0]-Qs[...,1]*Qs[...,1]-Qs[...,2]*Qs[...,2]+Qs[...,3]*Qs[...,3]

    return Rs

def rigid_from_3_points(N, Ca, C, non_ideal=False, eps=1e-8):
    #N, Ca, C - [B,L, 3]
    #R - [B,L, 3, 3], det(R)=1, inv(R) = R.T, R is a rotation matrix
    B,L = N.shape[:2]
    
    v1 = C-Ca
    v2 = N-Ca
    e1 = v1/(torch.norm(v1, dim=-1, keepdim=True)+eps)
    u2 = v2-(torch.einsum('bli, bli -> bl', e1, v2)[...,None]*e1)
    e2 = u2/(torch.norm(u2, dim=-1, keepdim=True)+eps)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.cat([e1[...,None], e2[...,None], e3[...,None]], axis=-1) #[B,L,3,3] - rotation matrix
    
    if non_ideal:
        v2 = v2/(torch.norm(v2, dim=-1, keepdim=True)+eps)
        cosref = torch.clamp( torch.sum(e1*v2, dim=-1), min=-1.0, max=1.0) # cosine of current N-CA-C bond angle
        costgt = cos_ideal_NCAC.item()
        cos2del = torch.clamp( cosref*costgt + torch.sqrt((1-cosref*cosref)*(1-costgt*costgt)+eps), min=-1.0, max=1.0 )
        cosdel = torch.sqrt(0.5*(1+cos2del)+eps)
        sindel = torch.sign(costgt-cosref) * torch.sqrt(1-0.5*(1+cos2del)+eps)
        Rp = torch.eye(3, device=N.device).repeat(B,L,1,1)
        Rp[:,:,0,0] = cosdel
        Rp[:,:,0,1] = -sindel
        Rp[:,:,1,0] = sindel
        Rp[:,:,1,1] = cosdel
    
        R = torch.einsum('blij,bljk->blik', R,Rp)

    return R, Ca

def get_t(N, Ca, C, non_ideal=False, eps=1e-5):
    I,B,L=N.shape[:3]
    Rs,Ts = rigid_from_3_points(N.view(I*B,L,3), Ca.view(I*B,L,3), C.view(I*B,L,3), non_ideal=non_ideal, eps=eps)
    Rs = Rs.view(I,B,L,3,3)
    Ts = Ts.view(I,B,L,3)
    t = Ts[:,:,None] - Ts[:,:,:,None] # t[0,1] = residue 0 -> residue 1 vector
    return torch.einsum('iblkj, iblmk -> iblmj', Rs, t) # (I,B,L,L,3)

def FAPE_loss(pred, true, score_scales,  d_clamp=10.0, d_clamp_inter=30.0, A=10.0, gamma=1.0, eps=1e-6):
    '''
    Calculate Backbone FAPE loss from RosettaTTAFold
    https://github.com/uw-ipd/RoseTTAFold2/blob/main/network/loss.py
    Input:
        - pred: predicted coordinates (I, B, L, n_atom, 3)
        - true: true coordinates (B, L, n_atom, 3)
    Output: str loss
    '''
    I = pred.shape[0]
    true = true.unsqueeze(0)
    t_tilde_ij = get_t(true[:,:,:,0], true[:,:,:,1], true[:,:,:,2])
    t_ij = get_t(pred[:,:,:,0], pred[:,:,:,1], pred[:,:,:,2])

    difference = torch.sqrt(torch.square(t_tilde_ij-t_ij).sum(dim=-1) + eps)
    eij_label = difference[-1].clone().detach()

    clamp = torch.zeros_like(difference)

    # intra vs inter#me coded
    clamp[:,True] = d_clamp

    difference = torch.clamp(difference, max=clamp)
    loss = difference / A # (I, B, L, L)

    # calculate masked loss (ignore missing regions when calculate loss)
    loss = (loss[:,True]).sum(dim=-1) / (torch.ones_like(loss).sum()+eps) # (I)
    #loss = loss.sum(dim=-1) #sum to batch dimension weight score by t-value (varies per batch) #me

    # weighting loss
    w_loss = torch.pow(torch.full((I,), gamma, device=pred.device), torch.arange(I, device=pred.device))
    w_loss = torch.flip(w_loss, (0,))
    w_loss = w_loss / w_loss.sum()

    #w_loss = score_scales.to(loss.device)#me
    #w_loss = w_loss / w_loss.sum() #do i need to normalize score_scales?#me

    tot_loss = (w_loss * loss).sum()
#     tot_loss = loss.sum()#me
    
    return tot_loss, loss.detach()