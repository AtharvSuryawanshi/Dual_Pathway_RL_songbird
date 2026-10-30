import json 
import os
from functions import remove_prev_files
from model import NN
from env import build_and_run

# load parameters from json file
params_path = "params.json"
# Open the file and read the contents
with open(params_path, "r") as f: 
    parameters = json.load(f)
# running conditions
# N_SYLL = parameters['run']['N_SYLL']
ANNEALING = parameters['params']['ANNEALING']

# if N_SYLL > 5 or N_SYLL < 1:
#     ValueError('Invalid number of syllables')
RANDOM_SEED = 20
remove_prev_files()
print(build_and_run(RANDOM_SEED, annealing = ANNEALING, plot = True, parameters = parameters, NN = NN))
# build_and_run(RANDOM_SEED, annealing = ANNEALING, plot = True, parameters = parameters1, NN = NN)