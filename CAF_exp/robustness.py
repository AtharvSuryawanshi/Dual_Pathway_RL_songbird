import json 
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import core
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
from functions import remove_prev_files
from model import NN
from Dual_Pathway_RL_songbird.CAF_exp.env_lite import build_and_run
from functions import find_neighboring_directories
import time 

NOS_SEEDS = 50
time_per_iter = 5.5
state = 5
np.random.seed(state)
seeds = np.random.randint(0, 100000, NOS_SEEDS)
seeds.sort()
wanted_directories = ["BG_INTACT_DAYS"] #["BG_NOISE", "RA_NOISE", "LEARNING_RATE_RL", "REWARD_WINDOW", "LEARNING_RATE_HL","TARGET_WIDTH","ANNEALING", "JUMP_MID", "JUMP_SLOPE", "JUMP_FACTOR", "RA_SIG_SLOPE", "balance_factor"]#["ANNEALING", "BG_NOISE", "LEARNING_RATE_HL", "LEARNING_RATE_RL", "RA_NOISE", "N_BG_CLUSTERS", "N_DISTRACTORS", "REWARD_WINDOW", "TARGET_WIDTH"]                                       
neighboring_directories = find_neighboring_directories()
for directory in neighboring_directories:
    if directory in wanted_directories:
        print(f"Directory: {directory}")
        np_path1 = f"{directory}/overall_returns.npy"
        np_path2 = f"{directory}/parameter_values.npy"
        np_file_name1 = os.path.basename(np_path1)
        np_file_name2 = os.path.basename(np_path2)
        if os.path.isfile(np_path1) and np_file_name1.endswith(".npy"):
            os.remove(np_path1)
            print(f"Deleted NumPy file: {np_path1}")
        if os.path.isfile(np_path2) and np_file_name2.endswith(".npy"):
            os.remove(np_path2)
            print(f"Deleted NumPy file: {np_path2}")

total_parameters = 0
for directory in neighboring_directories:
    if directory in wanted_directories:
        # load parameters from json file
        nos_parameters = 0
        print(f"Seeds: {seeds}")
        for potential_filename in os.listdir(directory):
            if potential_filename.startswith("parameters_") and potential_filename.endswith(".json"):
                total_parameters += 1

print(f"Total number of parameters: {total_parameters}")
time_remaining_in_s = time_per_iter * total_parameters * NOS_SEEDS
time_remaining = np.round(time_remaining_in_s / 60, 2)
print(f"Time remaining: {time_remaining} minutes")

start_time = time.perf_counter()

for directory in neighboring_directories:
    if directory in wanted_directories:
        # load parameters from json file
        nos_parameters = 0
        # print(f"Seeds: {seeds}")
        for potential_filename in os.listdir(directory):
            if potential_filename.startswith("parameters_") and potential_filename.endswith(".json"):
                nos_parameters += 1
        print(f"Number of parameters: {nos_parameters} for directory {directory}")

        overall_returns_b4cutoff = np.zeros((NOS_SEEDS, nos_parameters))
        overall_returns_aftercutoff = np.zeros((NOS_SEEDS, nos_parameters))
        overall_returns_end = np.zeros((NOS_SEEDS, nos_parameters))
        parameter_values = np.zeros(nos_parameters)
        j = 0
        for potential_filename in os.listdir(directory):
            if potential_filename.startswith("parameters_") and potential_filename.endswith(".json"):
                print(f"Potential filename: {potential_filename} with index {j}")     
                if j >= nos_parameters:
                    print(f"Skipping file {potential_filename} as index {j} exceeds nos_parameters {nos_parameters}")
                    continue
                param = potential_filename.split("_")[1].split(".jso")[0]
                full_filename = os.path.join(directory, potential_filename)
                # load parameters from json file
                with open(full_filename, "r") as f:
                    parameters = json.load(f)
                    N_SYLL = parameters['params']['N_SYLL']
                    if N_SYLL != 1:
                        raise ValueError('nos syllables needs to be 1')
                    print(f"Opening JSON file: {full_filename}")
                    returns_b4cutoff = np.zeros((NOS_SEEDS))
                    returns_aftercutoff = np.zeros((NOS_SEEDS))
                    returns_end = np.zeros((NOS_SEEDS))
                    for i, seed in enumerate(seeds):
                        elapsed_time = time.perf_counter() - start_time
                        annealing_val = parameters['params']['ANNEALING']
                        returns_b4cutoff[i], returns_aftercutoff[i], returns_end[i] = build_and_run(seed, annealing=annealing_val, plot=False, parameters=parameters, NN=NN)
                        print(F"Seed: {seed} with index {i} and returns: {returns_b4cutoff[i]}, {returns_aftercutoff[i]}, {returns_end[i]}")
                        print(f"Time remaining now: {np.round((time_remaining_in_s - elapsed_time) / 60, 2)} minutes")
                    overall_returns_b4cutoff[:, j] = returns_b4cutoff
                    overall_returns_aftercutoff[:, j] = returns_aftercutoff
                    overall_returns_end[:, j] = returns_end
                    # overall_returns_cutoff[:, j] = returns_cutoff
                    # overall_returns_nocutoff[:, j] = returns_nocutoff
                    parameter_values[j] = param
                    j += 1
        np.save(f"{directory}/overall_returns_b4cutoff.npy", overall_returns_b4cutoff)
        np.save(f"{directory}/overall_returns_aftercutoff.npy", overall_returns_aftercutoff)
        np.save(f"{directory}/overall_returns_end.npy", overall_returns_end)
        np.save(f"{directory}/parameter_values.npy", parameter_values)