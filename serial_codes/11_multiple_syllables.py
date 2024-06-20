# addition of syllables
# i died writing this code

import os
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap

# save plots in a folder
save_dir = "plots"
def remove_prev_files():
    os.makedirs(save_dir, exist_ok = True)
    for filename in os.listdir(save_dir):
        os.remove(os.path.join(save_dir, filename))

# 2D reward landscapesno
def gaussian(coordinates, height, mean, spread):
    ''' Returns a scalar value for given coordinates in a 2D gaussian distribution'''
    x, y = coordinates[0], coordinates[1]
    return height * np.exp(-((x-mean[0])**2 + (y-mean[1])**2)/(2*spread**2))

def new_sigmoid(x, m=0.0, a=0.0):
    """ Returns an output between -1 and 1 """
    return (2 / (1 + np.exp(-1*(x-a)*m))) - 1

def sigmoid(x, m =0.0 , a=0.0 ):
    """ Returns an output between 0 and 1 """
    return 1 / (1 + np.exp(-1*(x-a)*m))

# syllables
N_SYLL = 2
if N_SYLL > 5:
    ValueError('Number of syllables should be less than 5')
TRIALS = 1000
DAYS = 61 # 60 days of learning and 1 day of testing

RANDOM_SEED = 21 #np.random.randint(0, 1000)
print(f'Random seed is {RANDOM_SEED}')
np.random.seed(RANDOM_SEED)

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
RA_sig_slope = 18 # most steep such that MC output is not skewed
RA_sig_mid = 0
# Sigmoid on MC is removed
# MC_sig_slope = 1 # 5 if lesser -> more difficult to climb the hill, assymptotes before 
# MC_sig_mid = 0

# parameters
reward_window = 10
BG_noise = 0.1

# Run paraneters

N_DISTRACTORS = 10
LEARING_RATE_RL = 0.1
LEARNING_RATE_HL = 2e-5 # small increase compared to CODE_8


# modes
ANNEALING = True
ANNEALING_SLOPE = 4 
ANNEALING_MID = 2
HEBBIAN_LEARNING = True
balance_factor = 2
BG_influence = True

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
        self.bg_influence = BG_influence
            
    def forward(self, hvc_array):
        self.hvc = hvc_array
        # count number of 1 in hvc, divide bg by that number
        num_ones = np.count_nonzero(hvc_array == 1)
        self.bg = new_sigmoid(np.dot(hvc_array/num_ones, self.W_hvc_bg) + np.random.normal(0, BG_noise, self.bg_size), m = BG_sig_slope, a = BG_sig_mid)
        self.ra = new_sigmoid(np.dot(self.bg, self.W_bg_ra/np.sum(self.W_bg_ra, axis=0)) * balance_factor * self.bg_influence + np.dot(hvc_array/num_ones, self.W_hvc_ra)* HEBBIAN_LEARNING, m = RA_sig_slope, a = RA_sig_mid) 
        ''' even after BG cut off, output should remain still the same'''
        # self.mc = new_sigmoid(np.dot(self.ra, self.W_ra_mc/np.sum(self.W_ra_mc, axis=0)), m = MC_sig_slope, a = MC_sig_mid)
        # self.bg = np.dot(hvc_array/num_ones, self.W_hvc_bg)  #outputs to +-0.98
        # self.ra = np.dot(self.bg, self.W_bg_ra/np.sum(self.W_bg_ra, axis=0)) * balance_factor  + np.dot(hvc_array/num_ones, self.W_hvc_ra)* HEBBIAN_LEARNING #outputs to +-0.40
        self.mc = np.dot(self.ra, self.W_ra_mc/np.sum(self.W_ra_mc, axis=0)) # outputs to +-0.50
        return self.mc, self.ra, self.bg

class Environment:
    def __init__(self, hvc_size, bg_size, ra_size, mc_size, seed):
        self.hvc_size = hvc_size
        self.bg_size = bg_size
        self.ra_size = ra_size
        self.mc_size = mc_size
        self.seed = seed
        self.model = NN(hvc_size, bg_size, ra_size, mc_size)
        self.centers = np.random.uniform(-0.9, 0.9, (N_SYLL, 2))
        self.heights = np.random.uniform(0.2, 0.7, (N_SYLL, N_DISTRACTORS))
        self.means = np.random.uniform(-1, 1, (N_SYLL,N_DISTRACTORS, 2))
        self.spreads = np.random.uniform(0.1, 0.6, (N_SYLL, N_DISTRACTORS))
        self.rewards = np.zeros((N_SYLL, TRIALS*DAYS))
        self.actions = np.zeros((N_SYLL, TRIALS*DAYS, self.mc_size))
        self.hvc_bg_array = np.zeros((N_SYLL, TRIALS*DAYS))
        self.bg_out = np.zeros((N_SYLL, TRIALS*DAYS))
        self.hvc_ra_array = np.zeros((N_SYLL, TRIALS*DAYS))
        self.ra_out = np.zeros((N_SYLL, TRIALS*DAYS))
        self.dw_day_array = np.zeros((N_SYLL, DAYS))
        self.pot_array = np.zeros((N_SYLL, DAYS))
        
    def get_reward(self, coordinates, syll):
        center = self.centers[syll, :]
        reward_scape = gaussian(coordinates, 1, center, 0.3)
        if N_DISTRACTORS == 0:
            return reward_scape
        hills = []
        hills.append(reward_scape)
        for i in range(N_DISTRACTORS):
            height = self.heights[syll, i]
            mean = self.means[syll, i,:]
            spread = self.spreads[syll, i]
            hills.append(gaussian(coordinates, height, mean, spread))
        return np.maximum.reduce(hills)
     
    def run(self, learning_rate, learning_rate_hl, annealing = False):
        self.annealing = annealing
        for syll in range(N_SYLL):
            self.model.bg_influence = True
            input_hvc = np.zeros(HVC_SIZE)
            input_hvc[syll] = 1
            rewards = np.zeros((DAYS*TRIALS))
            actions = np.zeros((DAYS*TRIALS, self.mc_size))
            dw_day_array = np.zeros((DAYS))
            pot_array = np.zeros((DAYS))
            index = 0
            for day in tqdm(range(DAYS)):
                dw_day = 0
                if day >= 10:
                    self.model.bg_influence = False
                for iter in range(TRIALS):
                    # reward and baseline
                    action, ra, bg = self.model.forward(input_hvc)
                    reward = self.get_reward(action, syll)
                    rewards[index] = reward
                    actions[index, :] = action
                    if index < reward_window:
                        reward_baseline = 0
                    else:
                        reward_baseline = np.mean(rewards[index-reward_window:index])
                    # Updates 
                    # RL update
                    dw_hvc_bg = learning_rate*(reward - reward_baseline)*input_hvc.reshape(self.hvc_size,1)*self.model.bg * self.model.bg_influence # RL update
                    self.model.W_hvc_bg += dw_hvc_bg
                    # HL update
                    dw_hvc_ra = learning_rate_hl*input_hvc.reshape(self.hvc_size,1)*self.model.ra*HEBBIAN_LEARNING # lr is supposed to be much smaller here
                    self.model.W_hvc_ra += dw_hvc_ra
                    # bound weights between +-1
                    self.model.W_hvc_bg = np.clip(self.model.W_hvc_bg, -1, 1)
                    self.model.W_hvc_ra = np.clip(self.model.W_hvc_ra, -1, 1)
                    dw_day += np.mean(np.abs(dw_hvc_bg))
                    self.hvc_bg_array[syll, index] = self.model.W_hvc_bg[syll,1]
                    self.bg_out[syll, index] = bg[1]
                    self.hvc_ra_array[syll, index] = self.model.W_hvc_ra[syll,1]
                    self.ra_out[syll, index] = ra[1]
                    # tqdm.write(f'Day: {day}, Iteration: {iter}, Reward: {reward}, Reward Baseline: {reward_baseline}')
                    index += 1
                if day % 1 == 0:   
                    tqdm.write(f'Day: {day}, Action: {action}, Reward: {reward}, Reward Baseline: {reward_baseline}')  
                # Annealing
                if annealing:
                    ''' input daily sum, output scaling factor for potentiation'''
                    p = dw_day*100
                    dw_day_array[day] = p
                    p = 1* sigmoid(1*p, m = ANNEALING_SLOPE, a = ANNEALING_MID)
                    potentiation_factor = np.zeros((self.hvc_size))
                    pot_array[day] = 1 - p
                    potentiation_factor[syll] = 1-p 
                    
                    night_noise = np.random.uniform(-1, 1, self.bg_size) # make it lognormal
                    dw_night = LEARING_RATE_RL*potentiation_factor.reshape(self.hvc_size,1)*night_noise*10*self.model.bg_influence
                    self.model.W_hvc_bg += dw_night
                    self.model.W_hvc_bg = (self.model.W_hvc_bg + 1) % 2 -1 # bound between -1 and 1 in cyclical manner
            self.dw_day_array[syll, :] = dw_day_array
            self.pot_array[syll, :] = pot_array
            self.rewards[syll, :] = rewards
            self.actions[syll, :] = actions
                
    # def plot_trajectory(self):
    #     fig, axs = plt.subplots(figsize=(10, 9))
    #     # generate grid 
    #     x, y = np.linspace(-1, 1, 50), np.linspace(-1, 1, 50)
    #     X, Y = np.meshgrid(x, y)
    #     Z = self.get_reward([X, Y])
        
    #     # Plot contour
    #     cmap = LinearSegmentedColormap.from_list('white_to_green', ['white', 'green'])
    #     contour = axs.contourf(X, Y, Z, levels=10, cmap=cmap)
    #     fig.colorbar(contour, ax=axs, label='Reward')
        
    #     # plot trajectory
    #     x_traj, y_traj = zip(*self.actions)
    #     axs.plot(x_traj[::10], y_traj[::10], 'r', label='Agent Trajectory', alpha = 0.5, linewidth = 0.5) # Plot every 20th point for efficiency
    #     axs.scatter(x_traj[0], y_traj[0], s=20, c='b', label='Starting Point')  # Plot first point as red circle
    #     axs.scatter(x_traj[-5:], y_traj[-5:], s=20, c='r', marker='x', label='Ending Point') # type: ignore
    #     axs.scatter(CENTER[0], CENTER[1], s=20, c='y', marker='x', label='target')  # type: ignore
    #     # labels
    #     axs.set_title('Contour plot of reward function')
    #     axs.set_xlabel('x')
    #     axs.set_ylabel('y')
    #     axs.legend()
    #     plt.tight_layout()
    #     plt.show()
        
    # def plot_results(self):
    #     fig, axs = plt.subplots(6, 1, figsize=(10, 15))
    #     axs[0].plot(self.rewards, '.', markersize=1, linestyle='None')
    #     axs[0].hlines(0.7, 0, len(self.rewards), colors='r', linestyles='dashed')
    #     axs[0].set_ylim(0, 1)
    #     axs[0].set_ylabel('Reward')
    #     axs[1].plot(self.hvc_bg_array)
    #     axs[1].set_ylim(-1, 1)
    #     axs[1].set_ylabel('HVC BG weights')
    #     axs[2].plot(self.bg_out,'.', markersize=0.5, linestyle='None')
    #     axs[2].set_ylim(-1, 1)
    #     axs[2].set_ylabel('BG output')
    #     axs[3].plot(self.hvc_ra_array)
    #     axs[3].set_ylim(-1, 1)
    #     axs[3].set_ylabel('HVC RA weights')
    #     axs[4].plot(self.actions)
    #     axs[4].plot(self.center[0]*np.ones(TRIALS*DAYS))
    #     axs[4].plot(self.center[1]*np.ones(TRIALS*DAYS))
    #     axs[4].legend(['x target', 'y target'])
    #     axs[4].set_ylabel('Motor Output')
    #     axs[4].set_ylim(-1, 1)
    #     axs[5].plot(self.ra_out)
    #     axs[5].set_ylim(-1, 1)
    #     axs[5].set_ylabel('RA activity')
    #     fig.suptitle('Results', fontsize=20)
    #     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #     plt.show()
    
    # def plot_dw_day(self):
    #     if ANNEALING:
    #         # Expand dw_day_array and pot_array to match the size of rewards
    #         expanded_dw_day_array = np.repeat(self.dw_day_array, len(self.rewards) // len(self.dw_day_array))
    #         expanded_pot_array = np.repeat(self.pot_array, len(self.rewards) // len(self.pot_array))
    #         plt.title('Annealing')
    #         plt.plot(expanded_dw_day_array, markersize=1, label='dW_day')
    #         plt.plot(expanded_pot_array, markersize=1, label='Potentiation factor')
    #         plt.plot(self.rewards, '.', markersize=1, label='Reward')
    #         plt.xlabel('Days')
    #         plt.ylabel('dW_day')
    #         plt.legend()
    #         plt.show()
        
    def save_trajectory(self, syll):
        fig, axs = plt.subplots(figsize=(10, 9))
        # generate grid 
        x, y = np.linspace(-1, 1, 50), np.linspace(-1, 1, 50)
        X, Y = np.meshgrid(x, y)
        Z = self.get_reward([X, Y], syll)
        
        # Plot contour
        cmap = LinearSegmentedColormap.from_list('white_to_green', ['white', 'green'])
        contour = axs.contourf(X, Y, Z, levels=10, cmap=cmap)
        fig.colorbar(contour, ax=axs, label='Reward')
        
        # plot trajectory
        x_traj, y_traj = zip(*self.actions[syll, :])
        axs.plot(x_traj[::10], y_traj[::10], 'r', label='Agent Trajectory', alpha = 0.5, linewidth = 0.5) # Plot every 20th point for efficiency
        axs.scatter(x_traj[0], y_traj[0], s=20, c='b', label='Starting Point')  # Plot first point as red circle
        axs.scatter(x_traj[-5:], y_traj[-5:], s=20, c='r', marker='x', label='Ending Point') # type: ignore
        axs.scatter(self.centers[syll, 0], self.centers[syll, 1], s=20, c='y', marker='x', label='target')  # type: ignore
        # labels
        axs.set_title('Contour plot of reward function')
        axs.set_xlabel('x')
        axs.set_ylabel('y')
        axs.legend()
        plt.tight_layout()
        # Create the "plots" directory if it doesn't exist
        os.makedirs(save_dir, exist_ok = True)
        # Clear previous plots (optional):
        # for filename in os.listdir(save_dir):
        #     if filename.startswith("trajectory") and filename.endswith(".png") or filename.endswith(".jpg"):
        #         os.remove(os.path.join(save_dir, filename))
        # Save the plot
        plt.savefig(os.path.join(save_dir, f"trajectory_{RANDOM_SEED}_{syll}.png"))
        plt.close()  # Close the plot to avoid memory leaks
        
    def save_results(self, syll):
        fig, axs = plt.subplots(6, 1, figsize=(10, 15))
        axs[0].plot(self.rewards[syll, :], '.', markersize=1, linestyle='None')
        axs[0].hlines(0.7, 0, len(self.rewards[syll, :]), colors='r', linestyles='dashed')
        axs[0].set_ylim(0, 1)
        axs[0].set_ylabel('Reward')
        axs[1].plot(self.hvc_bg_array[syll, :])
        axs[1].set_ylim(-1, 1)
        axs[1].set_ylabel('HVC BG weights')
        axs[2].plot(self.bg_out[syll, :],'.', markersize=0.5, linestyle='None')
        axs[2].set_ylim(-1, 1)
        axs[2].set_ylabel('BG output')
        axs[3].plot(self.hvc_ra_array[syll, :])
        axs[3].set_ylim(-1, 1)
        axs[3].set_ylabel('HVC RA weights')
        axs[4].plot(self.actions[syll, :])
        axs[4].plot(self.centers[syll, 0]*np.ones(TRIALS*DAYS))
        axs[4].plot(self.centers[syll, 0]*np.ones(TRIALS*DAYS))
        axs[4].legend(['x target', 'y target'])
        axs[4].set_ylabel('Motor Output')
        axs[4].set_ylim(-1, 1)
        axs[5].plot(self.ra_out[syll, :])
        axs[5].set_ylim(-1, 1)
        axs[5].set_ylabel('RA activity')
        fig.suptitle('Results', fontsize=20)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # Create the "plots" directory if it doesn't exist
        os.makedirs(save_dir, exist_ok = True)
        # Clear previous plots (optional):
        # for filename in os.listdir(save_dir):
        #     if filename.startswith("results") and filename.endswith(".png") or filename.endswith(".jpg"):
        #         os.remove(os.path.join(save_dir, filename))
        # Save the plot
        plt.savefig(os.path.join(save_dir, f"results_{RANDOM_SEED}_{syll}.png"))
        plt.close()  # Close the plot to avoid memory leaks
        
    def save_dw_day(self, syll):
        if ANNEALING:
            # Expand dw_day_array and pot_array to match the size of rewards
            expanded_dw_day_array = np.repeat(self.dw_day_array[syll, :], len(self.rewards[syll, :]) // len(self.dw_day_array[syll, :]))
            expanded_pot_array = np.repeat(self.pot_array[syll, :], len(self.rewards[syll, :]) // len(self.pot_array[syll, :]))
            plt.title('Annealing')
            plt.plot(expanded_dw_day_array, markersize=1, label='dW_day')
            plt.plot(expanded_pot_array, markersize=1, label='Potentiation factor')
            plt.plot(self.rewards[syll, :], '.', markersize=1, label='Reward', alpha = 0.1)
            plt.xlabel('Days')
            plt.ylabel('dW_day')
            plt.legend()
            # Create the "plots" directory if it doesn't exist
            os.makedirs(save_dir, exist_ok = True)
            # # Clear previous plots (optional):
            # for filename in os.listdir(save_dir):
            #     if filename.startswith("dw") and filename.endswith(".png") or filename.endswith(".jpg"):
            #         os.remove(os.path.join(save_dir, filename))
            # Save the plot
            plt.savefig(os.path.join(save_dir, f"dw_{RANDOM_SEED}_{syll}.png"))
            plt.close()  # Close the plot to avoid memory leaks
    
env = Environment(HVC_SIZE, BG_SIZE, RA_SIZE, MC_SIZE, RANDOM_SEED)
env.run(LEARING_RATE_RL, LEARNING_RATE_HL, ANNEALING)
remove_prev_files()
for i in range(N_SYLL):
    env.save_trajectory(i)
    env.save_results(i)
    if ANNEALING:
        env.save_dw_day(i)

def build_and_run(seed, annealing, plot):
    tqdm.write(f" Random seed is {seed}")
    np.random.seed(seed)
    env = Environment(HVC_SIZE, BG_SIZE, RA_SIZE, MC_SIZE, RANDOM_SEED)
    env.run(LEARING_RATE_RL, LEARNING_RATE_HL, ANNEALING)
    remove_prev_files()
    for i in range(N_SYLL):
        env.save_trajectory(i)
        env.save_results(i)
        if ANNEALING:
            env.save_dw_day(i)
    return np.mean(env.rewards[:,-100:], axis=1) 


