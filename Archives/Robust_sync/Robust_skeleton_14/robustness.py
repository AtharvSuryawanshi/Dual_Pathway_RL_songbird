import json 
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import core
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
from functions import remove_prev_files
from model import NN
from env import build_and_run
from functions import find_neighboring_directories

NOS_SEEDS = 50
np.random.seed(3)
seeds = np.random.randint(0, 1000, NOS_SEEDS)
seeds.sort()
wanted_directories = ["ANNEALING_MID"]#["ANNEALING", "BG_NOISE", "LEARNING_RATE_HL", "LEARNING_RATE_RL", "RA_NOISE", "N_BG_CLUSTERS", "N_DISTRACTORS", "REWARD_WINDOW", "TARGET_WIDTH"]                                       
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

for directory in neighboring_directories:
    if directory in wanted_directories:
        # load parameters from json file
        nos_parameters = 0
        print(f"Seeds: {seeds}")
        for potential_filename in os.listdir(directory):
            if potential_filename.startswith("parameters_") and potential_filename.endswith(".json"):
                nos_parameters += 1
        print(f"Number of parameters: {nos_parameters} for directory {directory}")

        overall_returns_cutoff = np.zeros((NOS_SEEDS, nos_parameters))
        overall_returns_nocutoff = np.zeros((NOS_SEEDS, nos_parameters))
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
                    returns_cutoff = np.zeros((NOS_SEEDS))
                    returns_nocutoff = np.zeros((NOS_SEEDS))    
                    for i, seed in enumerate(seeds):
                        annealing_val = parameters['params']['ANNEALING']
                        returns_cutoff[i], returns_nocutoff[i] = build_and_run(seed, annealing=annealing_val, plot=False, parameters=parameters, NN=NN)

                    overall_returns_cutoff[:, j] = returns_cutoff
                    overall_returns_nocutoff[:, j] = returns_nocutoff
                    parameter_values[j] = param
                    j += 1

        np.save(f"{directory}/overall_returns_cutoff.npy", overall_returns_cutoff)
        np.save(f"{directory}/overall_returns_nocutoff.npy", overall_returns_nocutoff)
        np.save(f"{directory}/parameter_values.npy", parameter_values)