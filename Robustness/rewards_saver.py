from env_figs import Environment
from env_figs import build_and_run
from model import NN
import json
import os
import numpy as np

# load parameters from json file
params_path = "params.json"
# Open the file and read the contents
with open(params_path, "r") as f:
    parameters = json.load(f)

# if file exists, remove it 
if os.path.exists("all_rewards.npy"):
    os.remove("all_rewards.npy")


def run_robustness(state, nos_seeds, parameters, NN):
    np.random.seed(state)
    time_per_iter = 6
    N_SYLL = parameters['params']['N_SYLL']
    if N_SYLL != 1:
        raise ValueError('nos syllables needs to be 1')
    total_time = time_per_iter * nos_seeds
    print(f"Total time: {total_time/60} minutes")
    all_rewards = np.zeros((nos_seeds, 61, 1000))
    seeds = np.random.randint(0, 1000000, nos_seeds)
    for i, seed in enumerate(seeds):
        env = Environment(seed, parameters, NN)
        env.run(parameters, True)
        all_rewards[i, :, :] = env.rewards.squeeze()
        print(f"Seed: {seed} with rewards: {all_rewards[i, :, :]}")
        print(f"Time remaining: {np.round((total_time - (i+1) * time_per_iter) / 60, 2)} minutes")
    return all_rewards

nos_seeds = 10
state = 10
total_returns = run_robustness(state, nos_seeds, parameters, NN)

np.save("all_rewards.npy", total_returns)

