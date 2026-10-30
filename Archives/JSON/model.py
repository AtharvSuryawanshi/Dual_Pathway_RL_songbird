import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
import json
from functions import *

# # load parameters from json file
# params_path = "JSON/params.json"
# # Open the file and read the contents
# with open(params_path, "r") as f:
#     parameters = json.load(f)

# Model
class NN:
    def __init__(self, parameters, seed):
        # setting parameters
        np.random.seed(seed)
        self.hvc_size = parameters['const']['HVC_SIZE']
        self.bg_size = parameters['const']['BG_SIZE']
        self.ra_size = parameters['const']['RA_SIZE']
        self.mc_size = parameters['const']['MC_SIZE']
        self.n_ra_clusters = parameters['const']['N_RA_CLUSTERS']
        self.n_bg_clusters = parameters['params']['N_BG_CLUSTERS']
        LOG_NORMAL = parameters['params']['LOG_NORMAL']
        self.bg_influence = parameters['params']['BG_influence']

        if LOG_NORMAL: # except ra mc, they need to be uniform
            self.W_hvc_bg = sym_lognormal_samples(minimum = -1, maximum = 1, size = (self.hvc_size, self.bg_size)) # changing from -1 to 1 
            self.W_hvc_ra = np.zeros((self.hvc_size, self.ra_size)) # connections start from 0 and then increase
            self.W_bg_ra = lognormal_weight((self.bg_size, self.ra_size)) # const from 0 to 1
            # self.W_ra_mc = np.random.uniform(0, 1, (self.ra_size, self.mc_size)) # const from 0 to 1  
            self.W_ra_mc = lognormal_weight((self.ra_size, self.mc_size)) # const from 0 to 1
        else:
            self.W_hvc_bg = np.random.uniform(-1,1,(self.hvc_size, self.bg_size)) # changing from -1 to 1 
            self.W_hvc_ra = np.zeros((self.hvc_size, self.ra_size)) # connections start from 0 and then increase
            self.W_bg_ra = np.random.uniform(0, 1, (self.bg_size, self.ra_size)) # const from 0 to 1
            self.W_ra_mc = np.random.uniform(0, 1, (self.ra_size, self.mc_size)) # const from 0 to 1
        # Creating channels
        # channel from ra to mc
        for i in range(self.n_ra_clusters):
            segPath = np.diag(np.ones(self.n_ra_clusters, int))[i]
            self.W_ra_mc[i*self.ra_size//self.n_ra_clusters : (i+1)*self.ra_size//self.n_ra_clusters] *= segPath
        # channel from bg to ra such that motor cortex components are independent of each other
        for i in range(self.n_bg_clusters):
            segPath = np.diag(np.ones(self.n_bg_clusters, int))[i]
            self.W_bg_ra[i*self.bg_size//self.n_bg_clusters : (i+1)*self.bg_size//self.n_bg_clusters] *= [j for j in segPath for r in range(self.ra_size//self.n_bg_clusters)]

            
    def forward(self, hvc_array, parameters):
        BG_NOISE = parameters['params']['BG_NOISE']
        RA_NOISE = parameters['params']['RA_NOISE']
        BG_SIG_SLOPE = parameters['params']['BG_SIG_SLOPE']
        RA_SIG_SLOPE = parameters['params']['RA_SIG_SLOPE']
        BG_sig_MID = parameters['params']['BG_sig_MID']
        RA_sig_MID = parameters['params']['RA_sig_MID']
        HEBBIAN_LEARNING = parameters['params']['HEBBIAN_LEARNING']
        balance_factor = parameters['params']['balance_factor']
        # count number of 1 in hvc, divide bg by that number
        num_ones = np.count_nonzero(hvc_array == 1)
        self.bg = new_sigmoid(np.dot(hvc_array/num_ones, self.W_hvc_bg) + np.random.normal(0, BG_NOISE, self.bg_size), m = BG_SIG_SLOPE, a = BG_sig_MID)
        self.ra = new_sigmoid(np.dot(self.bg, self.W_bg_ra/np.sum(self.W_bg_ra, axis=0)) * balance_factor * self.bg_influence + np.dot(hvc_array/num_ones, self.W_hvc_ra)* HEBBIAN_LEARNING + np.random.normal(0, RA_NOISE, self.ra_size)* HEBBIAN_LEARNING, m = RA_SIG_SLOPE, a = RA_sig_MID) 
        self.mc = 1.5*np.dot(self.ra, self.W_ra_mc/np.sum(self.W_ra_mc, axis=0)) # outputs to +-0.50
        return self.mc, self.ra, self.bg

# nn = NN(parameters, 0)
# hvc_array = np.zeros(nn.hvc_size)
# hvc_array[0] = 1
# a,b,c = nn.forward(hvc_array, parameters)
# print(a, b, c)