import json 
from functions import remove_prev_files
from model import NN
from env import build_and_run

# load parameters from json file
params_path = "params.json"
# Open the file and read the contents
with open(params_path, "r") as f: 
    parameters = json.load(f)
# running conditions

RANDOM_SEED = 2
remove_prev_files()
print(build_and_run(RANDOM_SEED, plot = True, parameters = parameters, NN = NN))