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

NOS_SEEDS = 25

neighboring_directories = find_neighboring_directories()
for directory in neighboring_directories:
    print(f"Directory: {directory}")
    np_path = f"{directory}/overall_returns.npy"
    np_file_name = os.path.basename(np_path)
    if os.path.isfile(np_path) and np_file_name.endswith(".npy"):
        os.remove(np_path)
        print(f"Deleted NumPy file: {np_path}")

for directory in neighboring_directories:
    seeds = np.random.randint(0, 1000, NOS_SEEDS)
    seeds.sort()
    # load parameters from json file
    nos_parameters = 0
    print(f"Seeds: {seeds}")
    for potential_filename in os.listdir(directory):
        if potential_filename.startswith("parameters_") and potential_filename.endswith(".json"):
            nos_parameters += 1
    print(f"Number of parameters: {nos_parameters}")
    overall_returns = np.zeros((NOS_SEEDS, nos_parameters))
    parameter_values = np.zeros(nos_parameters)
    for j, potential_filename in enumerate(os.listdir(directory)):
        if potential_filename.startswith("parameters_") and potential_filename.endswith(".json"):
            param = potential_filename.split("_")[1].split(".")[0]
            full_filename = os.path.join(directory, potential_filename)
            # load parameters from json file
            with open(full_filename, "r") as f:
                parameters = json.load(f)
                N_SYLL = parameters['run']['N_SYLL']
                if N_SYLL != 1:
                    ValueError('nos syllables needs to be 1')
                print(f"Opening JSON file: {full_filename}")
                returns = np.zeros((NOS_SEEDS))
                for i, seed in enumerate(seeds):
                    returns[i] = build_and_run(seed, annealing = True, plot = False, parameters = parameters, NN = NN)
                overall_returns[:, j] = returns
                parameter_values[j] = param
    np.save(f"{directory}/overall_returns.npy", overall_returns)
    np.save(f"{directory}/parameter_values.npy", parameter_values)
