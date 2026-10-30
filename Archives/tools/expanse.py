import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

'''Finding the distribution of NN models'''
def new_sigmoid(x, m=0.0, a=0.0):
    """ Returns an output between -1 and 1 """
    return (2 / (1 + np.exp(-1*(x-a)*m))) - 1

RANDOM_SEED = 42 #np.random.randint(0, 1000)
# np.random.seed(RANDOM_SEED)
CENTER = np.random.uniform(-0.9, 0.9, 2)
# layer sizes
HVC_SIZE = 100
BG_SIZE = 50
RA_SIZE = 100 
MC_SIZE = 2
N_RA_CLUSTERS = MC_SIZE
N_BG_CLUSTERS = 2

# sigmoid layer parameters
BG_sig_slope = 2.5  # uniform output 
BG_sig_mid = 0
RA_sig_slope = 9 # most steep such that MC output is not skewed
RA_sig_mid = 0
# Sigmoid on MC is removed
# MC_sig_slope = 1 # 5 if lesser -> more difficult to climb the hill, assymptotes before 
# MC_sig_mid = 0

# parameters
reward_window = 10
input = np.zeros(HVC_SIZE)
input[1] = 1
BG_noise = 0.1

# modes
HEBBIAN_LEARNING = False
balance_factor = 2

# Model
class NN:
    def __init__(self, hvc_size, bg_size, ra_size, mc_size):
        self.W_hvc_bg = np.random.uniform(-1, 1, (hvc_size, bg_size)) # changing from -1 to 1 
        self.W_hvc_ra = np.zeros((hvc_size, ra_size)) # connections start from 0 and then increase
        self.W_bg_ra = np.random.uniform(0, 1, (bg_size, ra_size)) # const from 0 to 1
        self.W_ra_mc = np.random.uniform(0, 1, (ra_size, mc_size)) # const from 0 to 1
        # channel from ra to mc
        for i in range(N_RA_CLUSTERS):
            segPath = np.diag(np.ones(N_RA_CLUSTERS, int))[i]
            self.W_ra_mc[i*ra_size//N_RA_CLUSTERS : (i+1)*ra_size//N_RA_CLUSTERS] *= segPath
        # channel from bg to ra such that motor cortex components are independent of each other
        for i in range(N_BG_CLUSTERS):
            segPath = np.diag(np.ones(N_BG_CLUSTERS, int))[i]
            self.W_bg_ra[i*bg_size//N_BG_CLUSTERS : (i+1)*bg_size//N_BG_CLUSTERS] *= [j for j in segPath for r in range(RA_SIZE//N_BG_CLUSTERS)]

        self.hvc_size = hvc_size
        self.bg_size = bg_size
        self.ra_size = ra_size
        self.mc_size = mc_size  
        self.ra_cluster_size = ra_size // N_RA_CLUSTERS
        self.bg_cluster_size = bg_size // N_BG_CLUSTERS
            
    def forward(self, hvc_array):
        self.hvc = hvc_array
        # count number of 1 in hvc, divide bg by that number
        num_ones = np.count_nonzero(hvc_array == 1)
        self.bg = new_sigmoid(np.dot(hvc_array/num_ones, self.W_hvc_bg) + np.random.normal(0, BG_noise, self.bg_size), m = BG_sig_slope, a = BG_sig_mid)
        self.ra = new_sigmoid(np.dot(self.bg, self.W_bg_ra/np.sum(self.W_bg_ra, axis=0)) * balance_factor  + np.dot(hvc_array/num_ones, self.W_hvc_ra)* HEBBIAN_LEARNING, m = RA_sig_slope, a = RA_sig_mid) 
        # self.mc = new_sigmoid(np.dot(self.ra, self.W_ra_mc/np.sum(self.W_ra_mc, axis=0)), m = MC_sig_slope, a = MC_sig_mid)
        # self.bg = np.dot(hvc_array/num_ones, self.W_hvc_bg)  #outputs to +-0.98
        # self.ra = np.dot(self.bg, self.W_bg_ra/np.sum(self.W_bg_ra, axis=0)) * balance_factor  + np.dot(hvc_array/num_ones, self.W_hvc_ra)* HEBBIAN_LEARNING #outputs to +-0.40
        self.mc = np.dot(self.ra, self.W_ra_mc/np.sum(self.W_ra_mc, axis=0)) # outputs to +-0.50
        return self.mc, self.ra, self.bg
    
class Environment:
    def __init__(self, hvc_size, bg_size, ra_size, mc_size):
        self.hvc_size = hvc_size
        self.bg_size = bg_size
        self.ra_size = ra_size
        self.mc_size = mc_size
        self.model = NN(hvc_size, bg_size, ra_size, mc_size)
    
    def plot_expanse(self, nos):
        # print(self.model.forward(input))
        plt.figure()
        x, y = np.zeros(nos), np.zeros(nos)
        for i in tqdm(range(nos)):
            self.model.W_hvc_bg = np.random.uniform(-1, 1, (self.hvc_size, self.bg_size))
            mc_out, _, __ = self.model.forward(input)
            x[i], y[i] = mc_out[0], mc_out[1]
        plt.hist2d(x,y, density=True, bins=30)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.colorbar()
        plt.title(f'MC histogram and RA slope:{RA_sig_slope}')
        plt.show()
    
    def plot_histogram_RA(self, nos):
        plt.figure()
        x = np.zeros((nos, self.ra_size))
        for i in tqdm(range(nos)):
            self.model.W_hvc_bg = np.random.uniform(-1, 1, (self.hvc_size, self.bg_size))
            _, x[i,:], __ = self.model.forward(input)
        plt.hist(x.flatten(), density=True)
        # plt.title(f'Histogram of inputs to RA')
        plt.title(f'Histogram sig(RA) with slope: {RA_sig_slope}')
        plt.show()
    
    def plot_histogram_BG(self, nos):
        plt.figure()
        x = np.zeros((nos, self.bg_size))
        for i in tqdm(range(nos)):
            self.model.W_hvc_bg = np.random.uniform(-1, 1, (self.hvc_size, self.bg_size))
            _, __,x[i,:] = self.model.forward(input)
        plt.hist(x.flatten(), density=True)
        # plt.title(f'Histogram of inputs to RA')
        plt.title(f'Histogram BG with slope:{BG_sig_slope}')
        plt.show()
        
    
env = Environment(HVC_SIZE, BG_SIZE, RA_SIZE, MC_SIZE)
# env.plot_histogram_BG(10000)
# env.plot_histogram_RA(10000)
env.plot_expanse(10000)