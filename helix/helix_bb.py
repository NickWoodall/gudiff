
import os
from math import cos,sin,tan,asin,acos,radians,sqrt,degrees,atan,atan2,copysign
import numpy as np

import pandas as pd
import pickle
import scipy
from scipy.stats import norm
import random
import time
import timeit
import math
import localization as lx
import gzip

import helix.util.npose_util as nu
import datetime


import joblib
import argparse


#reference straight helix, global
zero_ih = nu.npose_from_file('helix/util/zero_ih.pdb')
tt = zero_ih.reshape(int(len(zero_ih)/5),5,4)
stub = tt[7:10].reshape(15,4)

    
#####MODDED EP Recon Init to input numpy array instead of saved npz file, might have broken something
##### Added class method to fix
class EP_Recon():
    """Parrallel Class to BatchRecon to loading endpoints from elsewhere and run LoopedEndpoints"""
    
    def __init__(self, endpoints_in):
        self.endpoints_list = endpoints_in.copy()
    
    @classmethod 
    def from_np_file(cls, fname):

        rr = np.load(f'{fname}.npz', allow_pickle=True)
        endpoints_list = [rr[f] for f in rr.files][0]
        return cls(endpoints_list)
    
    def to_npose(self):

        def normalize(v):
            norm = np.linalg.norm(v)
            if norm == 0: 
                return v
            return v / norm

        def angle_two_vectors(v1,v2):
            #assuming normalize
            #https://onlinemschool.com/math/library/vector/angl/

            # cos α = 	a·b
            #         |a|·|b|

            dp = np.dot(v1,v2)
            return  acos(dp)
        
        def ep_to_xform(p1,p2):
            zUnit  = np.array([0,0,-1])
            vector = normalize(p2 - p1)

            axisRot = normalize(np.cross(vector,zUnit))
            ang = angle_two_vectors(vector, zUnit)


            length = int(np.round(np.linalg.norm(p2-p1)/1.51,decimals=4)) #why zero here? quick mod to 4
            halfLen = int(length/2)
            aRot=np.hstack((axisRot,[1]))

            mp = p1+vector*float(halfLen/length)*np.linalg.norm(p2-p1)


            #global variable
            len_zero_ih = int(len(zero_ih)/5)

            hLen = int((len_zero_ih-length)/2)
            xform1 = nu.xform_from_axis_angle_rad(aRot,-ang)
            xform1[0][3] = mp[0] 
            xform1[1][3] = mp[1] 
            xform1[2][3] = mp[2]
            zI = np.copy(zero_ih)

            #aligned_pose = nu.xform_npose(xform1, zI)

            if length % 2 == 1:
                aligned_pose = nu.xform_npose(xform1, zI[(hLen*5):(-hLen*5)] )
            else:
                aligned_pose = nu.xform_npose(xform1, zI)
                aligned_pose = aligned_pose[((hLen)*5):(-(hLen+1)*5)]

            return aligned_pose, length
    
        
        self.npose_list = []
        self.helixLength_list = [] 
        
        
        for y in range(len(self.endpoints_list)):
            apList = np.array(np.empty((0,4), np.float32))
            #hardcoded 4
            self.helixLength_list.append([])
            for x in range(0,len(self.endpoints_list[y]),2):
                t,h_length = ep_to_xform(self.endpoints_list[y][x],self.endpoints_list[y][x+1])
                self.helixLength_list[y].append(h_length)
                apList = np.vstack((apList,t))
                
            self.npose_list.append(apList)
        return self.npose_list



def whole_prot_clash_check(npose,hList,threshold=2.85):
    """Checks for clashes between helices, given list of where in residues helices start and stop."""
    
    indexList = []
    curI = 0
    
    #get helical indices in npose form
    for ind,i in enumerate(hList):
        indexList.append(list(range(curI,curI+i*5)))
        curI += i*5
        
    fullSet = set(range(len(npose)))
    
    clashedCount = 0
    
    #remove 1 helix at a time and check it for clashing with the other three
    for ind,i in enumerate(indexList):
        build = list(fullSet.difference(set(i)))
        clashedCount += check_clash(npose[build],npose[i],threshold)
        
    return clashedCount
        
def check_clash(build_set, query, threshold=2.85):
    """Return True if new addition clashes with current set"""
    
    #if null, re
    if len(build_set) <= 5 or len(query) <= 5:
        return True
    query_set = query[5:]
    seq_buff = 5 # +1 from old clash check, should be fine
    if len(query_set) < seq_buff:
        seq_buff = len(query_set)
    elif len(build_set) < seq_buff:
        seq_buff = len(build_set)

    axa = scipy.spatial.distance.cdist(build_set,query_set)
    for i in range(seq_buff):
        for j in range(seq_buff-i):
            axa[-(i+1)][j] = threshold + 10 # moded from .1 here
            

    if np.min(axa) < threshold: # clash condition
        return True

    return False

def get_neighbor_2D(build):
    """Return 2D Neighbor Matrix, for slicing later"""
    
    pose = build.reshape(int(len(build)/5),5,4)

    ca_cb = pose[:,1:3,:3]
    conevect = (ca_cb[:,1] - ca_cb[:,0] )
    # conevect_lens = np.sqrt( np.sum( np.square( conevect ), axis=-1 ) )
    # for i in range(len(conevect)):
    #     conevect[i] /= conevect_lens[i]

    conevect /= 1.5

    maxx = 11.3
    max2 = maxx*maxx

    neighs = np.zeros((len(ca_cb),(len(ca_cb))))

    core = 0
    surf = 0

    summ = 0
    for i in range(len(ca_cb)):

        vect = ca_cb[:,0] - ca_cb[i,1]
        
        vect_length2 = np.sum( np.square( vect ), axis=-1 )

        ind = np.where((vect_length2 < max2) | (vect_length2 > 4))[0]
        vect_length = np.sqrt(vect_length2)

        vect = np.divide(vect,vect_length.reshape(-1,1))

        # bcov hack to make it ultra fast
        # linear fit to the above sigmoid
        dist_term = np.zeros(len(vect))

        for j in ind:
            if ( vect_length[j] < 7 ):
                dist_term[j] = 1
            elif (vect_length[j] > maxx ):
                dist_term[j] = 0
            else:
                dist_term[j] = -0.23 * vect_length[j] + 2.6

        angle_term = ( np.dot(vect, conevect[i] ) + 0.5 ) / 1.5

        for j in ind:
            if ( angle_term[j] < 0 ):
                angle_term[j] = 0
        neighs[i] = dist_term * np.square( angle_term )

    return neighs

def get_scn(sc_matrix, indices=None, percent_core = True):
    """Returns percent of residues that are in the core of the protein"""
    #core is defined as having greater than 5.2 summed from neighbor matrix
    
    if indices:
        indices = np.array(indices,dtype=np.int32)
        summed = np.sum(sc_matrix[indices], axis= -1)
    else:
        indices = np.array(list(range(len(sc_matrix))))
        summed = np.sum(sc_matrix,axis=-1)
        
    if percent_core:
        out = (summed > 5.2).sum() / len(indices)
    else:
        #av_scn
        out = np.mean(summed)
    
    return out




