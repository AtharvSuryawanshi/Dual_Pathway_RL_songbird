import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from functions import *

'''Dual pathway in 1D. This code runs well where we have a good enough grandient
Low reward in landscape is not enough to learn the gradient'''

def reward_fn3(x):  # gaussian reward landscape
    return np.exp(-(1*(x+0.8))**2)

reward_fn = reward_fn3 # define the reward function to use

# layer sizes
HVC_SIZE = 10
BG_SIZE = 5
RA_SIZE = 10 
MC_SIZE = 1

# sigmoid layer parameters
BG_sig_slope = 5  # lesser, slower the learning # BG sigmoidal should be as less steep as possible
BG_sig_mid = 0
RA_sig_slope = 30 # RA sigmoidal should be as steep as possible
RA_sig_mid = 0
MC_sig_slope = 10 # if lesser -> more difficult to climb the hill, assymptotes before 
MC_sig_mid = 0

# parameters
reward_window = 10
input = np.zeros(HVC_SIZE)
input[0] = 1

# Run paraneters
LEARING_RATE_RL = 0.01
LEARNING_RATE_HL = 0.0005
TRIALS = 50000

# modes
HEBBIAN_LEARNING = False

# Model
class NN:
    def __init__(self, hvc_size, bg_size, ra_size, mc_size):
        self.W_hvc_bg = np.random.uniform(-1, 1, (hvc_size, bg_size)) # changing from -1 to 1 
        self.W_hvc_ra = np.zeros((hvc_size, ra_size)) # connections start from 0 and then increase
        self.W_bg_ra = np.random.uniform(0, 1, (bg_size, ra_size)) # const from 0 to 1
        self.W_ra_mc = np.random.uniform(0, 1, (ra_size, mc_size)) # const from 0 to 1
        self.hvc_size = hvc_size
        self.bg_size = bg_size
        self.ra_size = ra_size
        self.mc_size = mc_size  
            
    def forward(self, hvc_array):
        self.hvc = hvc_array
        # count number of 1 in hvc, divide bg by that number
        num_ones = np.count_nonzero(hvc_array == 1)
        self.bg = new_sigmoid(np.dot(hvc_array/num_ones, self.W_hvc_bg) + np.random.normal(0, 0.05, self.bg_size), m = BG_sig_slope, a = BG_sig_mid)
        self.ra = new_sigmoid(np.dot(self.bg/self.bg_size, self.W_bg_ra) + np.dot(hvc_array/num_ones, self.W_hvc_ra)*HEBBIAN_LEARNING, m = RA_sig_slope, a = RA_sig_mid)
        self.ra += np.random.normal(0, 0.01, self.ra_size)  # ask why out of sigmoid 
        self.mc = new_sigmoid(np.dot(self.ra/self.ra_size, self.W_ra_mc), m = MC_sig_slope, a = MC_sig_mid)
        return self.mc

class Environment:
    def __init__(self, hvc_size, bg_size, ra_size, mc_size):
        self.hvc_size = hvc_size
        self.bg_size = bg_size
        self.ra_size = ra_size
        self.mc_size = mc_size
        self.model = NN(hvc_size, bg_size, ra_size, mc_size)
        self.rewards = []
        self.actions = []
        
    def get_reward(self, action):
        return reward_fn(action)
    
    def run(self, iterations, learning_rate, learning_rate_hl, input_hvc):
        for iter in tqdm(range(iterations)):
            # reward and baseline
            action = self.model.forward(input_hvc)
            reward = self.get_reward(action)
            self.rewards.append(reward)
            self.actions.append(action)
            reward_baseline = np.mean(self.rewards[-reward_window:])
            # Updates 
            # RL update
            dw_hvc_bg = learning_rate*(reward - reward_baseline)*input_hvc.reshape(self.hvc_size,1)*self.model.bg # RL update
            self.model.W_hvc_bg += dw_hvc_bg
            # HL update
            dw_hvc_ra = learning_rate_hl*input_hvc.reshape(self.hvc_size,1)*self.model.ra*HEBBIAN_LEARNING # lr is supposed to be much smaller here
            self.model.W_hvc_ra += dw_hvc_ra
            if iter % 1000 == 0:    
                tqdm.write(f'Iteration: {iter}, Action: {action}, Reward: {reward}, Reward Baseline: {reward_baseline}')     
                
    def plot_results(self):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        # First subplot
        axs[0].plot(self.rewards)
        axs[0].set_xlabel('Trial')
        axs[0].set_ylim(0,1)
        axs[0].set_ylabel('Rewards')
        axs[0].set_title('Learning Curve')  
        axs[0].set_ylim(0,1)

        # Second subplot
        axs[1].plot(self.actions)
        axs[1].set_xlabel('Trials')
        axs[1].set_ylabel('Value')
        axs[1].set_ylim(-1,1)
        axs[1].set_title('Actions vs Trials')

        plt.tight_layout() 
        plt.show()

env = Environment(HVC_SIZE, BG_SIZE, RA_SIZE, MC_SIZE)
plot_reward_fn(reward_fn)
env.run(TRIALS, LEARING_RATE_RL, LEARNING_RATE_HL, input)
env.plot_results()
