from gettext import find

import numpy as np
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from dual_pathway_model.functions import *
# from scipy.signal import find_peaks


# Parameters 
# Get path relative to this file
config_path = Path(__file__).parent / "params_SA.yaml"

with open(config_path, "r") as f:
    params_base_SA = yaml.safe_load(f)
    print("Base parameters loaded from params_SA.yaml")



# Model
class NN_SA:
    def __init__(self, parameters, seed):
        self.mc_size = int(parameters['const']['MC_SIZE'])
        # # setting parameters
        # # np.random.seed(seed)
        # self.hvc_size = int(parameters['const']['HVC_SIZE'])
        # self.bg_size = int(parameters['const']['BG_SIZE'])
        # self.ra_size = int(parameters['const']['RA_SIZE'])
        # self.mc_size = int(parameters['const']['MC_SIZE'])
        # self.n_ra_clusters = int(parameters['const']['N_RA_CLUSTERS'])
        # self.n_bg_clusters = int(parameters['params']['N_BG_CLUSTERS'])
        # LOG_NORMAL = parameters['params']['LOG_NORMAL']
        # self.bg_influence = parameters['params']['BG_influence']
        # self.LANDSCAPE = int(parameters['params']['LANDSCAPE'])
        # if self.LANDSCAPE == 0:
        #     self.limit = 1.5
        # else:
        #     self.limit = 1

        # if LOG_NORMAL:
        #     self.W_hvc_bg = sym_lognormal_samples(minimum = -1, maximum = 1, size = (self.hvc_size, self.bg_size)) # changing from -1 to 1 
        #     self.W_hvc_ra = np.zeros((self.hvc_size, self.ra_size)) # connections start from 0 and then increase
        #     self.W_bg_ra = lognormal_weight((self.bg_size, self.ra_size)) # const from 0 to 1
        #     # self.W_ra_mc = np.random.uniform(0, 1, (self.ra_size, self.mc_size)) # const from 0 to 1  
        #     self.W_ra_mc = lognormal_weight((self.ra_size, self.mc_size)) # const from 0 to 1
        # else:
        #     self.W_hvc_bg = np.random.uniform(-1,1,(self.hvc_size, self.bg_size)) # changing from -1 to 1 
        #     self.W_hvc_ra = np.zeros((self.hvc_size, self.ra_size)) # connections start from 0 and then increase
        #     self.W_bg_ra = np.random.uniform(0, 1, (self.bg_size, self.ra_size)) # const from 0 to 1
        #     self.W_ra_mc = np.random.uniform(0, 1, (self.ra_size, self.mc_size)) # const from 0 to 1
        # # Creating channels
        # # channel from ra to mc
        # for i in range(self.n_ra_clusters):
        #     segPath = np.diag(np.ones(self.n_ra_clusters, int))[i]
        #     self.W_ra_mc[i*self.ra_size//self.n_ra_clusters : (i+1)*self.ra_size//self.n_ra_clusters] *= segPath
        # # channel from bg to ra such that motor cortex components are independent of each other
        # for i in range(self.n_bg_clusters):
        #     segPath = np.diag(np.ones(self.n_bg_clusters, int))[i]
        #     self.W_bg_ra[i*self.bg_size//self.n_bg_clusters : (i+1)*self.bg_size//self.n_bg_clusters] *= [j for j in segPath for r in range(self.ra_size//self.n_bg_clusters)]

            
    def forward(self, action, noise, parameters, trial):
        return np.clip(action + np.random.normal(0, noise, self.mc_size), -1.5, 1.5)        
        # BG_NOISE = parameters['params']['BG_NOISE']
        # RA_NOISE = parameters['params']['RA_NOISE']
        # BG_NOISE_DECAY = parameters['params']['BG_NOISE_DECAY']
        # BG_SIG_SLOPE = parameters['params']['BG_SIG_SLOPE']
        # RA_SIG_SLOPE = parameters['params']['RA_SIG_SLOPE']
        # BG_sig_MID = parameters['params']['BG_sig_MID']
        # RA_sig_MID = parameters['params']['RA_sig_MID']
        # HEBBIAN_LEARNING = parameters['params']['HEBBIAN_LEARNING']
        # balance_factor = parameters['params']['balance_factor']
        # # count number of 1 in hvc, divide bg by that number
        # num_ones = np.count_nonzero(hvc_array == 1)
        # if trial is not None:
        #     BG_NOISE_size = BG_NOISE*np.exp(-trial*BG_NOISE_DECAY/60_000)
        # self.bg = new_sigmoid(np.dot(hvc_array/num_ones, self.W_hvc_bg) + np.random.normal(0, BG_NOISE_size, self.bg_size), m = BG_SIG_SLOPE, a = BG_sig_MID)
        # ra_noise = np.random.normal(0, RA_NOISE, self.ra_size)* HEBBIAN_LEARNING
        # self.ra = new_sigmoid(np.dot(self.bg, self.W_bg_ra/np.sum(self.W_bg_ra, axis=0)) * balance_factor * self.bg_influence + np.dot(hvc_array/num_ones, self.W_hvc_ra)* HEBBIAN_LEARNING + ra_noise, m = RA_SIG_SLOPE, a = RA_sig_MID)
        # self.mc = self.limit*np.dot(self.ra, self.W_ra_mc/np.sum(self.W_ra_mc, axis=0)) # outputs to +-0.50
        # self.ra_hvc = new_sigmoid(np.dot(hvc_array/num_ones, self.W_hvc_ra)* HEBBIAN_LEARNING + ra_noise, m = RA_SIG_SLOPE, a = RA_sig_MID)
        # self.mc_ra_hvc = self.limit*np.dot(self.ra_hvc, self.W_ra_mc/np.sum(self.W_ra_mc, axis=0))
        # self.mc_ra_bg = self.mc - self.mc_ra_hvc
        # return self.mc, self.ra, self.bg, self.mc_ra_bg

# nn = NN(parameters, 0)
# hvc_array = np.zeros(nn.hvc_size)
# hvc_array[0] = 1
# a,b,c, _ = nn.forward(hvc_array, parameters)
# print(a, b, c)

class Environment:
    def __init__(self, seed, parameters, NN_SA):
        # setting parameters
        self.DAYS = parameters['params']['DAYS']
        # self.BG_INTACT_DAYS = parameters['params']['BG_INTACT_DAYS']
        # self.HEARING_INTACT_DAYS = parameters['params']['HEARING_INTACT_DAYS']
        self.TRIALS = parameters['params']['TRIALS']
        self.N_SYLL = parameters['params']['N_SYLL']
        # self.hvc_size = parameters['const']['HVC_SIZE']
        # self.bg_size = parameters['const']['BG_SIZE']
        # self.ra_size = parameters['const']['RA_SIZE']
        self.mc_size = parameters['const']['MC_SIZE']
        self.LANDSCAPE = parameters['params']['LANDSCAPE']
        self.n_distractors = parameters['params']['N_DISTRACTORS']
        self.target_width = parameters['params']['TARGET_WIDTH']
        q = np.linspace(0, 10, self.DAYS * self.TRIALS + 1)[1:]
        self.temperature = parameters['params']['TEMPERATURE']*(1-np.exp(-1/q))
        # self.RECORD_WEIGHTS = parameters['params']['RECORD_WEIGHTS']
        self.seed = seed
        np.random.seed(seed)
        if self.LANDSCAPE == 0:
            self.limit = 1.5
        else:
            self.limit = 1
        self.model = NN_SA(parameters, seed)
        # landscape parameters
        if self.LANDSCAPE == 0: # ARTIFICAL LANDSCAPE
            n_distractors = int(self.n_distractors)
            self.centers = np.random.uniform(-0.9, 0.9, (self.N_SYLL, 2))
            self.heights = np.random.uniform(0.2, 0.7, (self.N_SYLL, n_distractors))
            self.means = np.random.uniform(-1, 1, (self.N_SYLL,n_distractors, 2))
            self.spreads = np.random.uniform(0.1, 0.6, (self.N_SYLL, n_distractors))
        else: # SYRINX LANDSCAPE
            if self.N_SYLL > 4:
                raise ValueError('Only 4 syllables are available in the syrinx landscape')
            self.syrinx_contours = []
            self.syrinx_targets = []
            for syll in range(self.N_SYLL):
                base = np.load(f"contours/Syll{syll+1}.npy")
                Z, target = make_contour(base)
                self.syrinx_contours.append(Z)
                self.syrinx_targets.append(target)
                self.centers = np.array(self.syrinx_targets)
        # data storage
        self.rewards = np.zeros((self.DAYS, self.TRIALS, self.N_SYLL))
        self.actions = np.zeros((self.DAYS, self.TRIALS, self.N_SYLL, self.mc_size))
        # self.actions_bg = np.zeros((self.DAYS, self.TRIALS, self.N_SYLL, self.mc_size))
        # self.hvc_bg_array = np.zeros((self.DAYS, self.TRIALS, self.N_SYLL))
        # self.hvc_ra_array = np.zeros((self.DAYS, self.TRIALS, self.N_SYLL))
        # if self.RECORD_WEIGHTS:
        #     self.hvc_bg_array_all = np.zeros((self.DAYS, self.TRIALS, self.N_SYLL, self.hvc_size, self.bg_size))   
        #     self.hvc_ra_array_all = np.zeros((self.DAYS, self.TRIALS, self.N_SYLL, self.hvc_size, self.ra_size)) 
        # self.hvc_bg_array_all = np.zeros((self.DAYS, self.TRIALS, self.N_SYLL, 8))   
        # self.hvc_ra_array_all = np.zeros((self.DAYS, self.TRIALS, self.N_SYLL, 8))
        
        # self.bg_out = np.zeros((self.DAYS, self.TRIALS, self.N_SYLL))
        # self.ra_out = np.zeros((self.DAYS, self.TRIALS, self.N_SYLL))

        # self.ra_all = np.zeros((self.DAYS, self.TRIALS,self.N_SYLL, self.ra_size))
        # self.bg_all = np.zeros((self.DAYS, self.TRIALS,self.N_SYLL, self.bg_size))
        # self.ra_all = np.zeros((self.DAYS, self.TRIALS,self.N_SYLL, 8))
        # self.bg_all = np.zeros((self.DAYS, self.TRIALS,self.N_SYLL, self.bg_size))
        # self.dw_day_array = np.zeros((self.DAYS, self.N_SYLL))
        # self.pot_array = np.zeros((self.DAYS, self.N_SYLL))
        # self.jump_size_array = np.zeros((self.DAYS, self.N_SYLL))
        # self.RPE = np.zeros((self.DAYS, self.TRIALS, self.N_SYLL)) 
        # self.RPE_SUM = np.zeros((self.DAYS, self.TRIALS, self.N_SYLL))
        # self.potentiation_factor_all = np.zeros((self.DAYS, self.N_SYLL, self.hvc_size, self.bg_size))
        # self.dist_from_target = np.zeros((self.DAYS, self.TRIALS, self.N_SYLL))

        
        
    def artificial_landscape(self, coordinates, syll):
        center = self.centers[syll, :]
        reward_scape = gaussian(coordinates, 1, center, self.target_width)
        if self.n_distractors == 0:
            return reward_scape
        hills = []
        hills.append(reward_scape)
        for i in range(int(self.n_distractors)):
            height = self.heights[syll, i]
            mean = self.means[syll, i,:]
            spread = self.spreads[syll, i]
            hills.append(gaussian(coordinates, height, mean, spread))
        return np.maximum.reduce(hills)
    
    def syrinx_landscape(self, coordinates, syll, n = 256):  
        contour = self.syrinx_contours[syll]
        target_pos = self.syrinx_targets[syll]
        x, y = coordinates[0], coordinates[1]
        x = max(min(x, 0.999), -1)
        y = max(min(y, 0.999), -1)
        x = int((x + 1) / 2 * n)
        y = int((y + 1) / 2 * n)
        return contour[x, y]

        
    def get_reward(self, coordinates, syll):
        # landscape creation and reward calculation
        if self.LANDSCAPE == False:
            return self.artificial_landscape(coordinates, syll)
        else:
            return self.syrinx_landscape(coordinates, syll)
             
    def run(self, parameters, annealing = False):
        # modes 
        # self.annealing = parameters['params']['ANNEALING']
        # self.model.bg_influence = True
        # # learning parameters
        # self.learning_rate = parameters['params']['LEARNING_RATE_RL']
        # learning_rate_hl = parameters['params']['LEARNING_RATE_HL']
        # REWARD_WINDOW = int(parameters['params']['REWARD_WINDOW'])
        # HEBBIAN_LEARNING = parameters['params']['HEBBIAN_LEARNING']
        # ANNEALING_SLOPE = parameters['params']['ANNEALING_SLOPE']
        # ANNEALING_MID = parameters['params']['ANNEALING_MID']
        # ANNEALING_MID_DECAY = parameters['params']['ANNEALING_MID_DECAY']
        # WEIGHT_JUMP = parameters['params']['WEIGHT_JUMP']
        # JUMP_MID = parameters['params']['JUMP_MID']
        # JUMP_SLOPE = parameters['params']['JUMP_SLOPE']
        # JUMP_FACTOR = parameters['params']['JUMP_FACTOR']
        # HARD_BOUND = parameters['params']['HARD_BOUND']
        prev_reward = 0
        noise = parameters['params']['NOISE']
        action = np.random.uniform(-1.5, 1.5, self.mc_size)
        # each day, 1000 trial, n_syll syllables
        for day in tqdm(range(self.DAYS)):
            # dw_day = np.zeros(self.N_SYLL)
            # self.model.bg_influence = True
            # if day >= self.BG_INTACT_DAYS:
            #     self.model.bg_influence = False # BG lesion on the last day
            # sum_RPE = np.zeros(self.N_SYLL)
            for iter in range(self.TRIALS):
                for syll in range(self.N_SYLL):
                    # input from HVC is determined by the syllable
                    # input_hvc = np.zeros(self.hvc_size)
                    # input_hvc[syll] = 1
                    # reward, action and baseline
                    action_potential = self.model.forward(action, noise, parameters, day*self.TRIALS + iter)
                    reward_potential = self.get_reward(action_potential, syll)
                    difference_reward = reward_potential - prev_reward
                    acceptance_probability = np.exp(difference_reward/self.temperature[iter])
                    if difference_reward > 0 or np.random.uniform(0,1) < acceptance_probability:
                        reward = reward_potential
                        action = action_potential
                    # print(f"Day: {day}, Diff: {difference_reward:.2f} Temperature: {self.Temperature[iter]:.3f} Ratio: {difference_reward/self.Temperature[iter]:.2f} Prob: {acceptance_probability:.2f}")
                    prev_reward = reward
                    self.rewards[day, iter, syll] = reward
                    self.actions[day, iter, syll,:] = action

            #         if day < self.HEARING_INTACT_DAYS:
            #             reward = self.get_reward(action, syll)
            #             reward_baseline = 0
            #             if iter < REWARD_WINDOW and iter > 0:
            #                 reward_baseline = np.mean(self.rewards[day, :iter, syll])
            #             elif iter >= REWARD_WINDOW:
            #                 reward_baseline = np.mean(self.rewards[day, iter-REWARD_WINDOW:iter, syll])
            #         else: # hearing removed
            #             reward = 0
            #             reward_baseline = -1
            #         # saving updates
            #         self.rewards[day, iter, syll] = reward
            #         self.actions[day, iter, syll,:] = action
            #         self.actions_bg[day, iter, syll,:] = action_bg
            #         # Updating weights
            #         # RL update
            #         dw_hvc_bg = self.learning_rate*(reward - reward_baseline)*input_hvc.reshape(self.hvc_size,1)*self.model.bg * self.model.bg_influence # RL update
            #         # self.model.W_hvc_bg += dw_hvc_bg
            #         # HL update
            #         dw_hvc_ra = learning_rate_hl*input_hvc.reshape(self.hvc_size,1)*self.model.ra*HEBBIAN_LEARNING # lr is supposed to be much smaller here
            #         # self.model.W_hvc_ra += dw_hvc_ra
            #         # bound weights between +-1
            #         # np.clip(self.model.W_hvc_bg, -1, 1, out = self.model.W_hvc_bg)
            #         # np.clip(self.model.W_hvc_ra, -1, 1, out = self.model.W_hvc_ra)
            #         if HARD_BOUND:
            #             self.model.W_hvc_bg += dw_hvc_bg
            #             self.model.W_hvc_ra += dw_hvc_ra    
            #             np.core.umath.maximum(np.core.umath.minimum(self.model.W_hvc_bg, 1, out = self.model.W_hvc_bg), -1, out = self.model.W_hvc_bg) # type: ignore
            #             np.core.umath.maximum(np.core.umath.minimum(self.model.W_hvc_ra, 1, out = self.model.W_hvc_ra), -1, out = self.model.W_hvc_ra) # type: ignore
            #         else: 
            #             self.model.W_hvc_bg += dw_hvc_bg*(1 - self.model.W_hvc_bg)*(self.model.W_hvc_bg + 1)
            #             self.model.W_hvc_ra += dw_hvc_ra*(1 - self.model.W_hvc_ra)*(self.model.W_hvc_ra + 1)
            #         # storing values for plotting
            #         self.RPE[day, iter, syll] = reward - reward_baseline   
            #         sum_RPE[syll] += self.RPE[day, iter, syll]
            #         self.RPE_SUM[day, iter, syll] = sum_RPE[syll]
            #         dw_day[syll] += np.mean(np.abs(dw_hvc_bg))
            #         self.hvc_bg_array[day, iter, syll] = self.model.W_hvc_bg[syll,1]
            #         self.bg_out[day, iter, syll] = bg[1]
            #         self.hvc_ra_array[day, iter, syll] = self.model.W_hvc_ra[syll,1]
            #         if self.RECORD_WEIGHTS:
            #             self.hvc_ra_array_all[day, iter, syll, :] = self.model.W_hvc_ra[syll,:]
            #             self.hvc_bg_array_all[day, iter, syll, :] = self.model.W_hvc_bg[syll,:]
            #         if iter == 0:
            #             hvc_bg_start = self.model.W_hvc_bg.copy()
            #         if iter == self.TRIALS-1:
            #             hvc_bg_end = self.model.W_hvc_bg.copy()
            #         self.ra_out[day, iter, syll] = ra[0]
            #         self.ra_all[day, iter, syll, :] = ra # np.concatenate([ra[:4], ra[-4:]])
            #         self.bg_all[day, iter, syll, :] = bg
            #         self.dist_from_target[day, iter, syll] = np.linalg.norm(action - self.centers[syll, :]) 

            # # Annealing
            # if self.annealing:
            #     for syll in range(self.N_SYLL):
            #         if WEIGHT_JUMP == 0:
            #             ''' input daily sum, output scaling factor for potentiation'''
            #             # OLD WAY OF DOING THINGS
            #             # calculating potentiation 
            #             d = dw_day[syll]*100 # scaling up to be comparable
            #             annealing_mid_final = ANNEALING_MID*np.exp(-ANNEALING_MID_DECAY*day/60)
            #             p = 1 * sigmoid(1*d, m = annealing_mid_final, a = ANNEALING_MID)
            #             potentiation_factor = np.zeros((self.hvc_size))
            #             potentiation_factor[syll] = 1-p 
            #             # implementing night weight changes
            #             night_noise = np.random.uniform(-1, 1, self.bg_size) # make it lognormal
            #             dw_night = self.learning_rate*potentiation_factor.reshape(self.hvc_size,1)*night_noise*3*self.model.bg_influence
            #             self.model.W_hvc_bg += dw_night
            #             self.model.W_hvc_bg = (self.model.W_hvc_bg + 1) % 2 -1 # bound between -1 and 1 in cyclical manner
            #             # storing values
            #             self.pot_array[day, syll] = 1-p
            #             self.dw_day_array[day, syll] = d
            #         elif WEIGHT_JUMP == 1:
            #             # NEW WAY OF DOING THINGS
            #             self.dw_day_array[day, syll] = self.RPE_SUM[day, iter, syll]
            #             # an alternate way of jumping! 
            #             rpe_sum_end_of_day = self.RPE_SUM[day, iter, syll]
            #             potentiation_factor = 1 - sigmoid(rpe_sum_end_of_day, m = JUMP_SLOPE, a = JUMP_MID)
            #             night_noise = np.random.uniform(-1, 1, (self.bg_size))   
            #             dw_night = self.learning_rate*potentiation_factor*night_noise*100*self.model.bg_influence
            #             W1 = self.model.W_hvc_bg[syll, :] + dw_night
            #             W2 = self.model.W_hvc_bg[syll, :] - dw_night
            #             indices_in_bound = (W1 <= 1) & (W1 >= -1)
            #             self.model.W_hvc_bg[syll, :] = W1*indices_in_bound + W2*(~indices_in_bound)
            #             x = self.model.W_hvc_bg
            #             self.model.W_hvc_bg[syll, :] += dw_night
            #             self.model.W_hvc_bg = (self.model.W_hvc_bg + 1) % 2 -1 # bound between -1 and 1 in cyclical manner
            #             diff = np.sum(np.abs(self.model.W_hvc_bg - x))
            #             self.jump_size_array[day, syll] = diff
            #             self.pot_array[day, syll] = potentiation_factor
            #         elif WEIGHT_JUMP == 2: # something Arthur told
            #             # print(hvc_bg_end[:, :].shape)
            #             abs_diff = np.abs(hvc_bg_end - hvc_bg_start)   # type: ignore
            #             # abs_diff = np.abs(self.hvc_bg_array_all[day, -1, syll, :] - self.hvc_bg_array_all[day, 0, syll, :])
            #             potentiation_factor = 1 - sigmoid(abs_diff, m = JUMP_SLOPE, a = JUMP_MID)
            #             night_noise = np.random.uniform(-1, 1, (self.bg_size))
            #             dw_night = self.learning_rate*JUMP_FACTOR*potentiation_factor[syll, :]*night_noise*self.model.bg_influence
            #             # print(np.max(dw_night), np.min(dw_night))
            #             W1 = self.model.W_hvc_bg[syll, :] + dw_night 
            #             W2 = self.model.W_hvc_bg[syll, :] - dw_night
            #             indices_in_bound = (W1 <= 1) & (W1 >= -1)
            #             self.pot_array[day, syll] = potentiation_factor[syll, 0]
            #             self.potentiation_factor_all[day, syll, :, :] = potentiation_factor
            #             self.model.W_hvc_bg[syll, :] = W1*indices_in_bound + W2*(~indices_in_bound)
            #             self.model.W_hvc_bg[syll, :] = np.clip(self.model.W_hvc_bg[syll, :], -1, 1)
            #             # if syll == 0 and day == 5:
            #             #     fig = plt.figure()
            #             #     plt.plot(abs_diff[0], alpha = .5)
            #             #     plt.plot(potentiation_factor[syll, :], alpha = .5)
            #             #     plt.plot(dw_night, alpha = .5)
            #             #     plt.legend(['abs diff', 'potentiation factor', 'dw night'])
            #             #     plt.ylim(-1,1)
            #             #     plt.show()
            #             #     fig = plt.figure()
            #             #     plt.scatter(abs_diff, potentiation_factor)
            #             #     plt.xlim(0,1)
            #             #     plt.ylim(0,1)
            #             #     plt.show()




    def save_trajectory(self, syll):
        fig, axs = plt.subplots(figsize=(10, 9))
        # generate grid 
        x, y = np.linspace(-self.limit,self.limit, 50), np.linspace(-self.limit, self.limit, 50)
        X, Y = np.meshgrid(x, y)
        Z = self.get_reward([X, Y], syll)
        # Plot contour
        cmap = LinearSegmentedColormap.from_list('white_to_green', ['white', 'black'])
        contour = axs.contourf(X, Y, Z, levels=10, cmap=cmap)
        fig.colorbar(contour, ax=axs, label='Reward')
        
        # plot trajectory
        x_traj, y_traj = zip(*self.actions[:,:, syll,:].reshape(-1, 2))
        axs.plot(x_traj[::10], y_traj[::10], 'yellow', label='Agent Trajectory', alpha = 0.5, marker = ".", linewidth = 0.1, markersize = 0.99) # Plot every 20th point for efficiency
        axs.scatter(x_traj[0], y_traj[0], s=100, c='blue', label='Starting Point', marker = 'x')  # type: ignore # Plot first point as red circle
        axs.scatter(x_traj[-5:], y_traj[-5:], s=100, c='r', marker='x', label='Ending Point') # type: ignore
        axs.scatter(self.centers[syll, 0], self.centers[syll, 1], s=100, c='green', marker='x', label='target')  # type: ignore
        # labels
        axs.set_title(f'Contour plot of reward function SEED:{self.seed} syllable: {syll}')
        axs.set_xlabel('x')
        axs.set_ylabel('y')
        axs.legend()
        plt.tight_layout()
        plt.show()
        # # Create the "plots" directory if it doesn't exist
        # os.makedirs(save_dir, exist_ok = True)
        # # Save the plot
        # plt.savefig(os.path.join(save_dir, f"trajectory_{self.seed}_{syll}.png"))
        # plt.close()  # Close the plot to avoid memory leaks
        
    def save_results(self, syll):
        fig, axs = plt.subplots(7, 1, figsize=(10, 10))
        axs[0].plot(self.rewards[:,:,syll].reshape(self.DAYS*self.TRIALS), '.', markersize=1, linestyle='None')
        axs[0].hlines(0.7, 0, self.DAYS*self.TRIALS, colors='r', linestyles='dashed')
        axs[0].set_ylim(0, 1)
        axs[0].set_ylabel('Reward')
        axs[1].plot(self.hvc_bg_array[:,:,syll].reshape(self.DAYS*self.TRIALS))
        axs[1].set_ylim(-1, 1)
        axs[1].set_ylabel('HVC BG weights')
        axs[2].plot(self.bg_out[:,:,syll].reshape(self.DAYS*self.TRIALS),'.', markersize=0.5, linestyle='None')
        axs[2].set_ylim(-1, 1)
        axs[2].set_ylabel('BG output')
        axs[3].plot(self.hvc_ra_array[:,:,syll].reshape(self.DAYS*self.TRIALS))
        axs[3].set_ylim(-1, 1)
        axs[3].set_ylabel('HVC RA weights')
        axs[4].plot(self.actions[:,:,syll,0].reshape(self.DAYS*self.TRIALS))
        axs[4].plot(self.actions[:,:,syll,1].reshape(self.DAYS*self.TRIALS))
        axs[4].plot(self.centers[syll, 0]*np.ones(self.TRIALS*self.DAYS))
        axs[4].plot(self.centers[syll, 1]*np.ones(self.TRIALS*self.DAYS))
        axs[4].legend(['x target', 'y target'])
        axs[4].set_ylabel('Motor Output')
        axs[4].set_ylim(-1, 1)
        axs[5].plot(self.ra_out[:,:,syll].reshape(self.DAYS*self.TRIALS))
        axs[5].set_ylim(-1, 1)
        axs[5].set_ylabel('RA activity')
        axs[5].set_xlabel('Days')
        axs[6].plot(self.RPE_SUM[:,:,syll].reshape(self.DAYS*self.TRIALS), '.', markersize=1, label='RPE_SUM', alpha = 0.1)
        axs[6].set_ylabel('RPE_SUM for a day')
        axs[6].set_ylim(-1, 4)
        for i in range(1,6):
            axs[i].set_xticks(range(0, self.DAYS*self.TRIALS, 10*self.TRIALS), range(0, self.DAYS, 10))
        fig.suptitle(f'Results SEED:{self.seed} syllable: {syll}', fontsize=20)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # type: ignore
        # plt.show()
        # # Create the "plots" directory if it doesn't exist
        # os.makedirs(save_dir, exist_ok = True)
        # # Save the plot
        # plt.savefig(os.path.join(save_dir, f"results_{self.seed}_{syll}.png"))
        # plt.close()  # Close the plot to avoid memory leaks
        
    def save_dw_day(self, syll):
        JUMP_MID = parameters['params']['JUMP_MID']
        if self.annealing:
            
            fig, axs = plt.subplots(3,1,figsize=(10, 5))
            expanded_dw_day_array = np.zeros((self.DAYS*self.TRIALS, self.N_SYLL)) 
            expanded_pot_array = np.zeros((self.DAYS*self.TRIALS, self.N_SYLL))
            # Expand dw_day_array and pot_array to match the size of rewards
            expanded_dw_day_array = np.repeat(self.dw_day_array[:, syll], self.DAYS*self.TRIALS// len(self.dw_day_array[:, syll]))
            expanded_dw_day = expanded_dw_day_array.reshape(self.DAYS*self.TRIALS)
            expanded_pot_array = np.repeat(self.pot_array[:, syll], self.DAYS*self.TRIALS// len(self.pot_array[:, syll]))
            expanded_jump_size_array = np.repeat(self.jump_size_array[:, syll], self.DAYS*self.TRIALS// len(self.jump_size_array[:, syll]))
            fig.suptitle(f'Annealing SEED:{self.seed} syllable: {syll}')
            # axs[0].plot(expanded_dw_day_array, markersize=1, label='dW_day')
            # axs[0].plot(ANNEALING_MID*np.ones((self.DAYS*self.TRIALS)), label = 'Threshold')
            # axs[0].set_ylabel('dW_day = mean(abs(dw_hvc_bg))')         
            # axs[0].set_ylim(0,4) 
            # axs[0].legend()
            axs[0].plot(expanded_pot_array, markersize=1, label='Potentiation factor')
            axs[0].plot(expanded_jump_size_array/50, markersize=1, label='Size of night jump') 
            axs[0].set_ylabel('Size of night jump')
            axs[0].legend()
            axs[0].set_ylim(0, 2)
            axs[1].plot(self.rewards[:,:,syll].reshape(self.DAYS*self.TRIALS), '.', markersize=1, label='Reward', alpha = 0.05)
            axs[1].set_ylabel('Rewards')
            axs[1].set_ylim(0, 1)
            axs[2].plot(self.RPE_SUM[:,:,syll].reshape(self.DAYS*self.TRIALS), '.', markersize=1, label='RPE_SUM', alpha = 0.1)
            axs[2].plot(JUMP_MID*np.ones((self.DAYS*self.TRIALS)))
            axs[2].plot(expanded_dw_day_array, markersize=1, label='End of day', color = 'brown')
            axs[2].set_ylabel('RPE sum for a day')
            axs[2].set_ylim(0, 1)    
            for i in range(3):
                axs[i].vlines(range(0, self.DAYS*self.TRIALS, self.TRIALS), -3, 10, colors='black', linestyles='dashed', alpha = 0.1)           
            plt.tight_layout()
            axs[2].legend()
            plt.show()
            # plt.savefig(os.path.join(save_dir, f"dw_day_{self.seed}_{syll}.png"))   
            # plt.close()

def plot_trajectory(obj, syll):
    fig, axs = plt.subplots(figsize=(10, 9))
    cmap = LinearSegmentedColormap.from_list('white_to_black', ['white', 'black'])
    if obj.LANDSCAPE == 0: # artificial landscape
        x_traj, y_traj = zip(*obj.actions[:,:, syll,:].reshape(-1, 2))
        limit = obj.limit
        print(limit)
        x, y = np.linspace(-limit, limit, 50), np.linspace(-limit, limit, 50)
        X, Y = np.meshgrid(x, y)
        Z = obj.get_reward([X, Y], syll)
        contour = axs.contourf(X, Y, Z, levels=10, cmap=cmap)
        fig.colorbar(contour, ax=axs, label='Reward')
        # # plot trajectory
        # axs.plot(x_traj[::10], y_traj[::10], 'yell ow', label='Agent Trajectory', alpha = 0.1, marker = ".", linewidth = 0.1, markersize = 0.99) # Plot every 20th point for efficiency
        # axs.scatter(x_traj[0], y_traj[0], s=100, c='blue', label='Starting Point', marker = 'x')  # type: ignore # Plot first point as red circle
        # axs.scatter(x_traj[-1001], y_traj[-1001], s=100, c='pink', marker='x', label='Before Lesion Ending Point')
        # axs.scatter(x_traj[-5:], y_traj[-5:], s=100, c='r', marker='x', label='After Leison Point') # type: ignore
        axs.scatter(obj.centers[syll, 0], obj.centers[syll, 1], s=100, c='green', marker='x', label='target')  # type: ignore
    else: 
        Z = obj.syrinx_contours[syll]
        target_pos = obj.syrinx_targets[syll]
        cs = plt.contourf(Z, cmap=cmap, extent=[-1, 1, -1, 1])
        fig.colorbar(cs, ax=axs, label='Reward')
        axs.scatter(target_pos[0], target_pos[1], s=100, c='green', marker='x', label='target')  # type: ignore
        # plot trajectory
    x_traj, y_traj = zip(*obj.actions[:,:, syll,:].reshape(-1, 2))
    axs.plot(x_traj[::10], y_traj[::10], 'yellow', label='Agent Trajectory', alpha = 0.5, linewidth = 0.1, marker='.', markersize = 0.99) # Plot every 20th point for efficiency
    axs.scatter(x_traj[0], y_traj[0], s=100, c='blue', label='Starting Point', marker = 'x')  # type: ignore # Plot first point as red circle
    axs.scatter(x_traj[-1001], y_traj[-1001], s=200, c='pink', marker='o', label='Before Lesion Ending Point')
    axs.scatter(x_traj[-1], y_traj[-1], s=200, c='r', marker='x', label='Ending Point') # type: ignore
    

    # labels
    axs.set_title(f'Contour plot of reward function SEED:{RANDOM_SEED} syllable: {syll}', fontsize = 15) # type: ignore 
    axs.set_ylabel(r'$P_{\alpha}$')
    axs.set_xlabel(r'$P_{\beta}$')
    axs.legend()
    plt.tight_layout()
    plt.show()
            
def build_and_run(seed, parameters, NN, lesion = False, 
                  output_reward= False, 
                  output_action= False,
                  plot = False,
                  find_nos_peaks = False):
    N_SYLL = parameters['params']['N_SYLL']
    DAYS = parameters['params']['DAYS']
    TRIALS = parameters['params']['TRIALS']
    # ANNEALING = parameters['params']['ANNEALING']
    # BG_INTACT_DAYS = parameters['params']['BG_INTACT_DAYS']
    tqdm.write(f" Random seed is {seed}")
    np.random.seed(seed)
    if plot:
        remove_prev_files()
    env = Environment(seed, parameters, NN)
    env.run(parameters)
    if find_nos_peaks:
        peaks = []
        for syll in range(N_SYLL):
            x, y = np.linspace(-env.limit, env.limit, 50), np.linspace(-env.limit, env.limit, 50)
            X, Y = np.meshgrid(x, y)
            Z = env.get_reward([X, Y], syll)
            rows, cols, heights = find_peaks_2d(Z, threshold=0.0)
            peaks.append(len(rows)) # only care about number of peaks, not their locations
    output = {}
    if output_reward:
        rewards_output = env.rewards[:,:,:] #.reshape(env.DAYS*env.TRIALS, env.N_SYLL)
        output['rewards'] = rewards_output
    if output_action:
        actions_output = env.actions[:,:,:,:] #.reshape(env.DAYS*env.TRIALS, env.N_SYLL, env.mc_size)
        output['actions'] = actions_output
    if output_reward or output_action:
        return output
    outputs = []
    for i in range(N_SYLL):
        if plot:
            env.save_trajectory(i)
            env.save_results(i)
            if ANNEALING:
                env.save_dw_day(i)
        rewards = env.rewards[:,:,i].reshape(env.DAYS*env.TRIALS)
        outputs.append(np.mean(rewards[-100:], axis=0))
        # return rewards after lesion and before lesion 
    if lesion: # terminal performance; before lesion; after lesion
        return np.mean(rewards[-100:], axis=0), np.mean(rewards[int((BG_INTACT_DAYS-1)*TRIALS-100):int((BG_INTACT_DAYS-1)*TRIALS)], axis=0), np.mean(rewards[int((BG_INTACT_DAYS+1)*TRIALS-100):int((BG_INTACT_DAYS+1)*TRIALS)], axis=0)
    else:
        if N_SYLL == 1:
            if find_nos_peaks:
                return outputs[0], peaks[0]
            else:
                return outputs[0]
        return outputs

