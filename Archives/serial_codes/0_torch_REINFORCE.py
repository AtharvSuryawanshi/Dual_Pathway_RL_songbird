import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
from tqdm import tqdm
'''
This is the code for learning a policy using REINFORCE algorithm implemented in pytorch.
The environment is a simple 1D environment where the agent has to learn to take actions
to maximize the reward. The reward function is predefined. 
'''

# We have 3 different reward functions to choose from. 
def reward_fn1(a): # a hill like reward landscape
    x = a - 0
    if x>=5 and x<6:
        return 1
    elif x<5 and x>0:
        return x/5
    elif x>5 and x<10:
        return 2.5 - x/4
    else:
        return 0

def reward_fn2(x): # a sine like reward landscape
    pi = np.pi
    x+=-0
    # return np.sin(x)
    if x>0 and x<pi:
        return np.sin(x)
    else:
        return 0

def reward_fn3(x):  # gaussian reward landscape
    return np.exp(-(x-5)**2)

reward_fn = reward_fn3 # define the reward function to use

def plot_reward_fn(): # reward plotting function
    x = np.linspace(-10, 10, 100)
    y = [reward_fn(i) for i in x]
    plt.plot(x, y)
    plt.xlabel('Action')
    plt.ylabel('Reward')
    plt.title('Reward Landscape')
    plt.show()
     
# Hyperparameters 
HIDDEN_SIZE = 10
INPUT_SIZE = 10
LEARNING_RATE = 0.001
TRIALS = 2000

# Define the neural network
class NN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NN, self).__init__()
        
        # Base layer is a simple feedforward layer
        self.base = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        # Mean is a single linear layer taking inputs from base layer
        self.mu = nn.Sequential(
            nn.Linear(hidden_size, 1),
        )
        # Variance is a single linear layer taking inputs from base layer
        self.var = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softplus()
        )
        self.value = nn.Linear(hidden_size, 1)
                
    def forward(self, x):
        base_out = self.base(x)
        return self.mu(base_out), self.var(base_out), self.value(base_out)
    
class Environment:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.model = NN(input_size, hidden_size)
        # arrays to store the values of mean, std dev, actions and rewards
        self.means = []
        self.stdevs = []
        self.actions = []
        self.rewards = []
    
    def get_input(self):
        return torch.rand(self.input_size)
    
    def run(self, learning_rate, trials):
        self.learning_rate = learning_rate
        self.trials = trials
        input = self.get_input() 
        optimizer = optim.Adam(self.model.parameters(), lr= self.learning_rate)
        prev_reward = 0
        for _ in tqdm(range(self.trials)):
            # input = self.get_input()    # Toggle this comment to make the input random
            mean, stdev, _ = self.model(input)
            optimizer.zero_grad()
            distribution = Normal(mean, 1*stdev)  # 5*stdev to make the initial exploration more
            action = distribution.sample()      # action is sampled from the policy of agent
            reward = reward_fn(action.item())   # reward is calculated for that using the reward function
            log_prob = distribution.log_prob(action)
            loss = -log_prob * (reward - prev_reward)  # REINFORCE loss
            loss.backward()
            optimizer.step()
            prev_reward = reward  
            self.means.append(mean.item())
            self.stdevs.append(stdev.item())
            self.actions.append(action.item())
            self.rewards.append(reward)
            
    def plot_results(self):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        # First subplot
        axs[0].plot(self.rewards)
        axs[0].set_xlabel('Trial')
        axs[0].set_ylabel('Rewards')
        axs[0].set_title('Learning Curve')
        # axs[0].set_ylim(0,1)

        # Second subplot
        axs[1].plot(self.means)
        axs[1].plot(self.stdevs)
        # axs[1].plot(actions)
        axs[1].set_xlabel('Trials')
        axs[1].legend(['Mean', 'Std Dev'])
        axs[1].set_ylabel('Value')
        axs[1].set_title('Mean and Std Dev vs Trials')

        plt.tight_layout() 
        plt.show()

env = Environment(INPUT_SIZE, HIDDEN_SIZE)
env.run(LEARNING_RATE, TRIALS)
plot_reward_fn()
env.plot_results()
