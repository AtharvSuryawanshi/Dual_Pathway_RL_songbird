# addition of syllables
# with log normal distributions for more biological plausibility

import os
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap

# save plots in a folder

folder_name = os.path.splitext(os.path.basename(__file__))[0]
# Create the folder path (assuming you want it in the current directory)
save_dir = f"{folder_name}_results"

def remove_prev_files(): # 
    '''removes previous files in the directory and creates one if it doesnt exist'''
    os.makedirs(save_dir, exist_ok = True)
    for filename in os.listdir(save_dir):
        os.remove(os.path.join(save_dir, filename))

# Basic functions
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

def sym_lognormal_samples(minimum, maximum, size, mu = 0.01, sigma = 0.5):
    """
    This function generates samples from a combined (original + reflected) lognormal distribution.
    Args:
        mu (float): Mean of the underlying normal distribution.
        sigma (float): Standard deviation of the underlying normal distribution.
        size (int): Number of samples to generate.
    Returns:
        numpy.ndarray: Array of samples from the combined lognormal distribution.
    """
    if size == 0:
        ValueError('Size cannot be zero')
    # Generate lognormal samples with half in one dimension only
    samples = np.random.lognormal(mu, sigma, size)
    combined_samples = np.concatenate((samples, samples * -1))/4
    # randomly remove samples such that size of combined_samples is equal to size
    combined_samples = np.random.choice(combined_samples.reshape(-1), size, replace = False)
    combined_samples = np.clip(combined_samples, minimum, maximum)
    return combined_samples

def lognormal_weight(size, mu = 0.01, sigma = 0.5):
    '''returns lognormal weights'''
    samples = np.random.lognormal(mu, sigma, size)/4
    samples = np.clip(samples, 0, 1)
    return samples

# running conditions
TRIALS = 1000
DAYS = 61 # 60 days of learning and 1 day of testing
N_SYLL = 1
if N_SYLL > 5 or N_SYLL < 1:
    ValueError('Invalid number of syllables')

# RANDOM_SEED = 43 #np.random.randint(0, 1000)
# print(f'Random seed is {RANDOM_SEED}')
# np.random.seed(RANDOM_SEED)

# modes
ANNEALING = True
ANNEALING_SLOPE = 4 
ANNEALING_MID = 2
HEBBIAN_LEARNING = True
LOG_NORMAL = True
balance_factor = 2
BG_influence = True

# parameters
REWARD_WINDOW = 10
BG_NOISE = 0.1

# Run paraneters
N_DISTRACTORS = 10
LEARING_RATE_RL = 0.1
LEARNING_RATE_HL = 2e-5 # small increase compared to CODE_8

# sigmoid layer parameters
BG_SIG_SLOPE = 2.5  # uniform output 
BG_sig_MID = 0
RA_SIG_SLOPE = 18 # most steep such that MC output is not skewed
RA_sig_MID = 0
# Sigmoid on MC is removed
# MC_SIG_SLOPE = 1 # 5 if lesser -> more difficult to climb the hill, assymptotes before 
# MC_sig_MID = 0

# layer sizes
HVC_SIZE = 100
BG_SIZE = 50
RA_SIZE = 100 
MC_SIZE = 2
N_RA_CLUSTERS = MC_SIZE
N_BG_CLUSTERS = 2

# Model
class NN:
    def __init__(self, hvc_size, bg_size, ra_size, mc_size):
        if LOG_NORMAL:
            self.W_hvc_bg = sym_lognormal_samples(minimum = -1, maximum = 1, size = (hvc_size, bg_size)) # changing from -1 to 1 
            self.W_hvc_ra = np.zeros((hvc_size, ra_size)) # connections start from 0 and then increase
            self.W_bg_ra = lognormal_weight((bg_size, ra_size)) # const from 0 to 1
            self.W_ra_mc = lognormal_weight((ra_size, mc_size)) # const from 0 to 1
        else:
            self.W_hvc_bg = np.random.uniform(-1,1,(hvc_size, bg_size)) # changing from -1 to 1 
            self.W_hvc_ra = np.zeros((hvc_size, ra_size)) # connections start from 0 and then increase
            self.W_bg_ra = np.random.uniform(0, 1, (bg_size, ra_size)) # const from 0 to 1
            self.W_ra_mc = np.random.uniform(0, 1, (ra_size, mc_size)) # const from 0 to 1
        # Creating channels
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
        self.bg = new_sigmoid(np.dot(hvc_array/num_ones, self.W_hvc_bg) + np.random.normal(0, BG_NOISE, self.bg_size), m = BG_SIG_SLOPE, a = BG_sig_MID)
        self.ra = new_sigmoid(np.dot(self.bg, self.W_bg_ra/np.sum(self.W_bg_ra, axis=0)) * balance_factor * self.bg_influence + np.dot(hvc_array/num_ones, self.W_hvc_ra)* HEBBIAN_LEARNING, m = RA_SIG_SLOPE, a = RA_sig_MID) 
        self.mc = np.dot(self.ra, self.W_ra_mc/np.sum(self.W_ra_mc, axis=0)) # outputs to +-0.50
        ''' even after BG cut off, output should remain still the same'''
        # below code is only for testing without sigmoidal functions
        # self.mc = new_sigmoid(np.dot(self.ra, self.W_ra_mc/np.sum(self.W_ra_mc, axis=0)), m = MC_SIG_SLOPE, a = MC_sig_MID)
        # self.bg = np.dot(hvc_array/num_ones, self.W_hvc_bg)  #outputs to +-0.98
        # self.ra = np.dot(self.bg, self.W_bg_ra/np.sum(self.W_bg_ra, axis=0)) * balance_factor  + np.dot(hvc_array/num_ones, self.W_hvc_ra)* HEBBIAN_LEARNING #outputs to +-0.40
        return self.mc, self.ra, self.bg

class Environment:
    def __init__(self, hvc_size, bg_size, ra_size, mc_size, seed):
        self.hvc_size = hvc_size
        self.bg_size = bg_size
        self.ra_size = ra_size
        self.mc_size = mc_size
        self.seed = seed
        self.model = NN(hvc_size, bg_size, ra_size, mc_size)
        # landscape parameters
        self.centers = np.random.uniform(-0.9, 0.9, (N_SYLL, 2))
        self.heights = np.random.uniform(0.2, 0.7, (N_SYLL, N_DISTRACTORS))
        self.means = np.random.uniform(-1, 1, (N_SYLL,N_DISTRACTORS, 2))
        self.spreads = np.random.uniform(0.1, 0.6, (N_SYLL, N_DISTRACTORS))
        # data storage
        self.rewards = np.zeros((DAYS, TRIALS, N_SYLL))
        self.actions = np.zeros((DAYS, TRIALS, N_SYLL, self.mc_size))
        self.hvc_bg_array = np.zeros((DAYS, TRIALS, N_SYLL))
        self.bg_out = np.zeros((DAYS, TRIALS, N_SYLL))
        self.hvc_ra_array = np.zeros((DAYS, TRIALS, N_SYLL))
        self.ra_out = np.zeros((DAYS, TRIALS, N_SYLL))
        self.dw_day_array = np.zeros((DAYS, N_SYLL))
        self.pot_array = np.zeros((DAYS, N_SYLL))
        
    def get_reward(self, coordinates, syll):
        # landscape creation and reward calculation
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
        # modes 
        self.annealing = annealing
        self.model.bg_influence = True
        # each day, 1000 trial, n_syll syllables
        for day in (range(DAYS)):
            dw_day = np.zeros(N_SYLL)
            self.model.bg_influence = True
            if day >= DAYS-1: 
                self.model.bg_influence = False # BG lesion on the last day
            for iter in range(TRIALS):
                for syll in range(N_SYLL):
                    # input from HVC is determined by the syllable
                    input_hvc = np.zeros(HVC_SIZE)
                    input_hvc[syll] = 1
                    # reward, action and baseline
                    action, ra, bg = self.model.forward(input_hvc)
                    reward = self.get_reward(action, syll)
                    self.rewards[day, iter, syll] = reward
                    self.actions[day, iter, syll,:] = action
                    reward_baseline = 0
                    if iter < REWARD_WINDOW and iter > 0:
                        reward_baseline = np.mean(self.rewards[day, :iter, syll])
                    elif iter >= REWARD_WINDOW:
                        reward_baseline = np.mean(self.rewards[day, iter-REWARD_WINDOW:iter, syll])
                    # Updating weights
                    # RL update
                    dw_hvc_bg = learning_rate*(reward - reward_baseline)*input_hvc.reshape(self.hvc_size,1)*self.model.bg * self.model.bg_influence # RL update
                    self.model.W_hvc_bg += dw_hvc_bg
                    # HL update
                    dw_hvc_ra = learning_rate_hl*input_hvc.reshape(self.hvc_size,1)*self.model.ra*HEBBIAN_LEARNING # lr is supposed to be much smaller here
                    self.model.W_hvc_ra += dw_hvc_ra
                    # bound weights between +-1
                    self.model.W_hvc_bg = np.clip(self.model.W_hvc_bg, -1, 1)
                    self.model.W_hvc_ra = np.clip(self.model.W_hvc_ra, -1, 1)
                    # storing values for plotting
                    dw_day[syll] += np.mean(np.abs(dw_hvc_bg))
                    self.hvc_bg_array[day, iter, syll] = self.model.W_hvc_bg[syll,1]
                    self.bg_out[day, iter, syll] = bg[1]
                    self.hvc_ra_array[day, iter, syll] = self.model.W_hvc_ra[syll,1]
                    self.ra_out[day, iter, syll] = ra[0]
            # if day % 1 == 0:   
            #     tqdm.write(f'Day: {day}, Action: {action}, Reward: {reward}, Reward Baseline: {reward_baseline}')  
            # Annealing
            if self.annealing:
                for syll in range(N_SYLL):
                    ''' input daily sum, output scaling factor for potentiation'''
                    # calculating potentiation 
                    d = dw_day[syll]*100 # scaling up to be comparable
                    p = 1 * sigmoid(1*d, m = ANNEALING_SLOPE, a = ANNEALING_MID)
                    potentiation_factor = np.zeros((self.hvc_size))
                    potentiation_factor[syll] = 1-p 
                    # implementing night weight changes
                    night_noise = np.random.uniform(-1, 1, self.bg_size) # make it lognormal
                    dw_night = LEARING_RATE_RL*potentiation_factor.reshape(self.hvc_size,1)*night_noise*10*self.model.bg_influence
                    self.model.W_hvc_bg += dw_night
                    self.model.W_hvc_bg = (self.model.W_hvc_bg + 1) % 2 -1 # bound between -1 and 1 in cyclical manner
                    # storing values
                    self.pot_array[day, syll] = 1-p
                    self.dw_day_array[day, syll] = d
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
        x_traj, y_traj = zip(*self.actions[:,:, syll,:].reshape(-1, 2))
        axs.plot(x_traj[::10], y_traj[::10], 'r', label='Agent Trajectory', alpha = 0.5, linewidth = 0.5) # Plot every 20th point for efficiency
        axs.scatter(x_traj[0], y_traj[0], s=20, c='b', label='Starting Point')  # Plot first point as red circle
        axs.scatter(x_traj[-5:], y_traj[-5:], s=20, c='r', marker='x', label='Ending Point') # type: ignore
        axs.scatter(self.centers[syll, 0], self.centers[syll, 1], s=20, c='y', marker='x', label='target')  # type: ignore
        # labels
        axs.set_title(f'Contour plot of reward function SEED:{self.seed} syllable: {syll}')
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
        plt.savefig(os.path.join(save_dir, f"trajectory_{self.seed}_{syll}.png"))
        plt.close()  # Close the plot to avoid memory leaks
        
    def save_results(self, syll):
        fig, axs = plt.subplots(6, 1, figsize=(10, 15))
        axs[0].plot(self.rewards[:,:,syll].reshape(DAYS*TRIALS), '.', markersize=1, linestyle='None')
        axs[0].hlines(0.7, 0, DAYS*TRIALS, colors='r', linestyles='dashed')
        axs[0].set_ylim(0, 1)
        axs[0].set_ylabel('Reward')
        axs[1].plot(self.hvc_bg_array[:,:,syll].reshape(DAYS*TRIALS))
        axs[1].set_ylim(-1, 1)
        axs[1].set_ylabel('HVC BG weights')
        axs[2].plot(self.bg_out[:,:,syll].reshape(DAYS*TRIALS),'.', markersize=0.5, linestyle='None')
        axs[2].set_ylim(-1, 1)
        axs[2].set_ylabel('BG output')
        axs[3].plot(self.hvc_ra_array[:,:,syll].reshape(DAYS*TRIALS))
        axs[3].set_ylim(-1, 1)
        axs[3].set_ylabel('HVC RA weights')
        axs[4].plot(self.actions[:,:,syll,0].reshape(DAYS*TRIALS))
        axs[4].plot(self.actions[:,:,syll,1].reshape(DAYS*TRIALS))
        axs[4].plot(self.centers[syll, 0]*np.ones(TRIALS*DAYS))
        axs[4].plot(self.centers[syll, 1]*np.ones(TRIALS*DAYS))
        axs[4].legend(['x target', 'y target'])
        axs[4].set_ylabel('Motor Output')
        axs[4].set_ylim(-1, 1)
        axs[5].plot(self.ra_out[:,:,syll].reshape(DAYS*TRIALS))
        axs[5].set_ylim(-1, 1)
        axs[5].set_ylabel('RA activity')
        fig.suptitle(f'Results SEED:{self.seed} syllable: {syll}', fontsize=20)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # Create the "plots" directory if it doesn't exist
        os.makedirs(save_dir, exist_ok = True)
        # Clear previous plots (optional):
        # for filename in os.listdir(save_dir):
        #     if filename.startswith("results") and filename.endswith(".png") or filename.endswith(".jpg"):
        #         os.remove(os.path.join(save_dir, filename))
        # Save the plot
        plt.savefig(os.path.join(save_dir, f"results_{self.seed}_{syll}.png"))
        plt.close()  # Close the plot to avoid memory leaks
        
    def save_dw_day(self, syll):
        if ANNEALING:
            expanded_dw_day_array = np.zeros((DAYS*TRIALS, N_SYLL))
            expanded_pot_array = np.zeros((DAYS*TRIALS, N_SYLL))
            # Expand dw_day_array and pot_array to match the size of rewards
            expanded_dw_day_array = np.repeat(self.dw_day_array[:, syll], DAYS*TRIALS// len(self.dw_day_array[:, syll]))
            expanded_pot_array = np.repeat(self.pot_array[:, syll], DAYS*TRIALS// len(self.pot_array[:, syll]))
            plt.title(f'Annealing SEED:{self.seed} syllable: {syll}')
            plt.plot(expanded_dw_day_array, markersize=1, label='dW_day')
            plt.plot(expanded_pot_array, markersize=1, label='Potentiation factor')
            plt.plot(self.rewards[:,:,syll].reshape(DAYS*TRIALS), '.', markersize=1, label='Reward', alpha = 0.1)
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
            plt.savefig(os.path.join(save_dir, f"dw_{self.seed}_{syll}.png"))
            plt.close()  # Close the plot to avoid memory leaks         


def build_and_run(seed, annealing, plot):
    tqdm.write(f" Random seed is {seed}")
    np.random.seed(seed)
    env = Environment(HVC_SIZE, BG_SIZE, RA_SIZE, MC_SIZE, seed)
    env.run(LEARING_RATE_RL, LEARNING_RATE_HL, annealing)
    returns = np.zeros(N_SYLL)
    for syll in (range(N_SYLL)):
        if plot:
            env.save_trajectory(syll)
            env.save_results(syll)
            if annealing:
                env.save_dw_day(syll)
        rewards = env.rewards[:,:,syll].reshape(DAYS*TRIALS)
        returns[syll] = np.mean(rewards[-100:], axis=0) 
    return returns

TRIALS = 1000
DAYS = 61
N_SYLL = 3
TEST_NOS = 20
seeds = np.random.randint(0,1000,size=TEST_NOS)
print("Seeds: ", seeds)
remove_prev_files()
returns1 = np.zeros((len(seeds), N_SYLL))
for i in tqdm(range(len(seeds))):
    seed = seeds[i]
    returns1[i, :] = build_and_run(seed, annealing=True, plot=False)
    tqdm.write(f"Seed: {seed}, Returns: {returns1[i, :]}")

LOG_NORMAL = False
returns2 = np.zeros((len(seeds), N_SYLL))
for i in tqdm(range(len(seeds))):
    seed = seeds[i]
    returns2[i, :] = build_and_run(seed, annealing=True, plot=False)
    tqdm.write(f"Seed: {seed}, Returns: {returns2[i, :]}")

# Save results to a text file
with open(os.path.join(save_dir, "results.txt"), "w") as f:
    f.write("Log-Normal Returns:\n")
    np.savetxt(f, returns1, fmt="%.4f")  # Save with 4 decimal places
    f.write("\nUniform Returns:\n")
    np.savetxt(f, returns2, fmt="%.4f")  # Save with 4 decimal places

# plot returns 1 vs 2
fig, ax = plt.subplots(1,1) 
for syll in range(N_SYLL):
    ax.plot(returns1[:, syll]*100, label=f'Log Normal {syll}', linewidth=0, marker='o')
    ax.plot(returns2[:, syll]*100, label=f'Uniform {syll}', linewidth=0, marker='+')
ax.set_ylim(0, 100)
plt.xticks(np.arange(0, len(seeds), step=1), labels=seeds)
ax.legend()
ax.set_xlabel('Seeds')
ax.set_ylabel('Performance %')
plt.hlines(70, 0, len(seeds), linestyles='dashed')
fig.suptitle('Results on weight distribtution', fontsize=20)
fig.set_size_inches(10,6)
# Save the plot
plt.savefig(os.path.join(save_dir, f"Overall_results.png"))
plt.close()  # Close the plot to avoid memory leaks
plt.show()