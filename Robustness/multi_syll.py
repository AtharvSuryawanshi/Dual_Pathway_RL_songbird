from env_lite import Environment
# from env import build_and_run
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
if os.path.exists("multi_syll_robust.npy"):
    os.remove("multi_syll_robust.npy")

def run_mutli_syll_robust(state, nos_seeds, parameters, NN):
    np.random.seed(state)
    N_SYLL = parameters['params']['N_SYLL']
    if N_SYLL == 1:
        raise Warning("Only one syllable, are you sure?")
    LANDSCAPE = parameters['params']['LANDSCAPE']
    # if LANDSCAPE == 0:
    #     raise Warning("LANDSCAPE is 0, are you sure?")  
    time_per_iter = 7
    total_time = time_per_iter * nos_seeds * N_SYLL
    print(f"Total time: {total_time/60} minutes")
    total_returns = np.zeros((nos_seeds, 2, 4))
    seeds = np.random.randint(0, 1000000, nos_seeds)
    for i, seed in enumerate(seeds):
        env = Environment(seed, parameters, NN)
        env.run(parameters, True)
        final_before_cut = env.rewards[59, -1, :]
        final_after_cut = env.rewards[60, -1, :]
        total_returns[i, 0, :] = final_before_cut
        total_returns[i, 1, :] = final_after_cut
        print(f"Seed: {seed} with final_before_cut: {final_before_cut} and final_after_cut: {final_after_cut}")
        print(f"Time remaining: {np.round((total_time - (i+1) * time_per_iter*N_SYLL) / 60, 2)} minutes")
    return total_returns

nos_seeds = 10
state = 5
total_returns = run_mutli_syll_robust(state, nos_seeds, parameters, NN)

np.save("multi_syll_robust.npy", total_returns)