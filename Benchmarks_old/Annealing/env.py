import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
from functions import *

class Environment:
    def __init__(self, seed, parameters, NN):
        # setting parameters
        self.DAYS = parameters['params']['DAYS']
        self.TRIALS = parameters['params']['TRIALS']
        self.N_SYLL = parameters['params']['N_SYLL']
        if self.N_SYLL != 1:
            raise ValueError("Only one syllable is allowed")
        self.out_size = parameters['const']['OUT_SIZE']
        self.n_distractors = parameters['params']['N_DISTRACTORS']
        self.target_width = parameters['params']['TARGET_WIDTH']
        self.seed = seed
        # np.random.seed(seed)
        self.model = NN(parameters, seed)
        # landscape parameters
        self.centers = np.random.uniform(-0.9, 0.9, (self.N_SYLL, 2))
        self.heights = np.random.uniform(0.2, 0.7, (self.N_SYLL, self.n_distractors))
        self.means = np.random.uniform(-1, 1, (self.N_SYLL,self.n_distractors, 2))
        self.spreads = np.random.uniform(0.1, 0.6, (self.N_SYLL, self.n_distractors))
        # data storage
        self.rewards = np.zeros((self.DAYS, self.TRIALS, self.N_SYLL))
        self.actions = np.zeros((self.DAYS, self.TRIALS, self.N_SYLL, self.out_size))
        q = np.linspace(0, 10, self.DAYS * self.TRIALS + 1)[1:]
        self.Temperature = parameters['params']['TEMPERATURE']*(1-np.exp(-1/q))
        
    def get_reward(self, coordinates, syll):
        # landscape creation and reward calculation
        center = self.centers[syll, :]
        reward_scape = gaussian(coordinates, 1, center, self.target_width)
        if self.n_distractors == 0:
            return reward_scape
        hills = []
        hills.append(reward_scape)
        for i in range(self.n_distractors):
            height = self.heights[syll, i]
            mean = self.means[syll, i,:]
            spread = self.spreads[syll, i]
            hills.append(gaussian(coordinates, height, mean, spread))
        return np.maximum.reduce(hills)
     
    def run(self, parameters):
        # learning parameters
        self.learning_rate = parameters['params']['LEARNING_RATE_RL']
        # each day, 1000 trial, n_syll syllables
        prev_reward = 0
        noise = parameters['params']['NOISE']
        action = np.random.uniform(-1.5, 1.5, self.out_size)
        iter = 0
        for day in tqdm(range(self.DAYS)):
            for trial in range(self.TRIALS):
                for syll in range(self.N_SYLL):
                    # reward, action and baseline
                    action_potential = self.model.forward(action, noise, parameters, iter)
                    reward_potential = self.get_reward(action_potential, syll)
                    difference_reward = reward_potential - prev_reward
                    acceptance_probability = np.exp(difference_reward/self.Temperature[iter])
                    if difference_reward > 0 or np.random.uniform(0,1) < acceptance_probability:
                        reward = reward_potential
                        action = action_potential
                    # print(f"Day: {day}, Diff: {difference_reward:.2f} Temperature: {self.Temperature[iter]:.3f} Ratio: {difference_reward/self.Temperature[iter]:.2f} Prob: {acceptance_probability:.2f}")
                    prev_reward = reward
                    self.rewards[day, trial, syll] = reward
                    self.actions[day, trial, syll,:] = action
                    iter += 1
        
    def save_trajectory(self, syll):
        fig, axs = plt.subplots(figsize=(10, 9))
        # generate grid 
        limit = 1.5
        x, y = np.linspace(-limit,limit, 50), np.linspace(-limit, limit, 50)
        X, Y = np.meshgrid(x, y)
        Z = self.get_reward([X, Y], syll)
        # Plot contour
        cmap = LinearSegmentedColormap.from_list('white_to_green', ['white', 'black'])
        contour = axs.contourf(X, Y, Z, levels=10, cmap=cmap)
        fig.colorbar(contour, ax=axs, label='Reward')
        
        # plot trajectory
        x_traj, y_traj = zip(*self.actions[:,:, syll,:].reshape(-1, 2))
        axs.plot(x_traj[::10], y_traj[::10], 'yellow', label='Agent Trajectory', alpha = 0.2, linewidth = 0, marker='.') # Plot every 20th point for efficiency
        axs.scatter(x_traj[0], y_traj[0], s=100, c='blue', label='Starting Point', marker = 'x')  # type: ignore # Plot first point as red circle
        axs.scatter(x_traj[-5:], y_traj[-5:], s=100, c='r', marker='x', label='Ending Point') # type: ignore
        axs.scatter(self.centers[syll, 0], self.centers[syll, 1], s=50, c='green', marker='x', label='target')  # type: ignore
        # labels
        axs.set_title(f'Contour plot of reward function SEED:{self.seed} syllable: {syll}')
        axs.set_ylabel(r'$P_{\alpha}$')
        axs.set_xlabel(r'$P_{\beta}$')
        axs.legend()
        plt.tight_layout()
        # Create the "plots" directory if it doesn't exist
        os.makedirs(save_dir, exist_ok = True)
        # Save the plot
        plt.savefig(os.path.join(save_dir, f"trajectory_{self.seed}_{syll}.png"))
        plt.close()  # Close the plot to avoid memory leaks
        
    def save_results(self, syll):
        fig, axs = plt.subplots(2, 1, figsize=(10, 15))
        axs[0].plot(self.rewards[:,:,syll].reshape(self.DAYS*self.TRIALS), '.', markersize=1, linestyle='None')
        axs[0].hlines(0.7, 0, self.DAYS*self.TRIALS, colors='r', linestyles='dashed')
        axs[0].set_ylim(0, 1)
        axs[0].set_ylabel('Reward')
        axs[1].plot(self.actions[:,:,syll,0].reshape(self.DAYS*self.TRIALS))
        axs[1].plot(self.actions[:,:,syll,1].reshape(self.DAYS*self.TRIALS))
        axs[1].plot(self.centers[syll, 0]*np.ones(self.TRIALS*self.DAYS))
        axs[1].plot(self.centers[syll, 1]*np.ones(self.TRIALS*self.DAYS))
        axs[1].legend(['x target', 'y target'])
        axs[1].set_ylabel('Motor Output')
        axs[1].set_ylim(-1, 1)
        axs[1].set_xlabel('Days')
        # for i in range(1,6):
            # axs[i].set_xticks(range(0, self.DAYS*self.TRIALS, 10*self.TRIALS), range(0, self.DAYS, 10))
        fig.suptitle(f'Results SEED:{self.seed} syllable: {syll}', fontsize=20)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # Create the "plots" directory if it doesn't exist
        os.makedirs(save_dir, exist_ok = True)
        # Save the plot
        plt.savefig(os.path.join(save_dir, f"results_{self.seed}_{syll}.png"))
        plt.close()  # Close the plot to avoid memory leaks

    def plot_combined_returns(self):
        plt.figure()
        for i in range(self.N_SYLL):
            plt.plot(self.rewards[:,:,i].reshape(self.DAYS*self.TRIALS), '.', markersize=0.3, color='pink')
            lined = np.convolve(self.rewards[:,:,i].reshape(self.DAYS*self.TRIALS), np.ones((100,))/100, mode='same')
            plt.plot(lined, color='violet')
        plt.plot(np.convolve(self.rewards.mean(axis = 2).reshape(self.DAYS*self.TRIALS),np.ones((100,))/100, mode='same'), color='black', label='Mean')
        plt.xlabel('Days')
        plt.xticks(range(0, self.DAYS*self.TRIALS, 10*self.TRIALS), range(0, self.DAYS, 10))
        plt.ylabel('Performance Metric')
        plt.legend()
        plt.show()

def build_and_run(seed, plot, parameters, NN):
    N_SYLL = parameters['params']['N_SYLL']
    if N_SYLL != 1:
        raise ValueError("Only one syllable is allowed")    
    tqdm.write(f" Random seed is {seed}")
    np.random.seed(seed)
    env = Environment(seed, parameters, NN)
    env.run(parameters)
    for i in range(N_SYLL):
        if plot:
            env.save_trajectory(i)
            env.save_results(i)
        rewards = env.rewards[:,:,0].reshape(env.DAYS*env.TRIALS)
    # env.plot_combined_returns()
    return np.mean(rewards[-100:], axis=0)


# load parameters from json file
# params_path = "Benchmarks/annealing/params.json"
# params_path = "params.json"
# Open the file and read the contents
# with open(params_path, "r") as f:
#     parameters = json.load(f)
# # running conditions
# seed = 41
# env = Environment(seed, parameters, NN)
# env.run(parameters)
# env.save_trajectory(0)
# env.save_results(0)