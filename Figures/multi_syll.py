import os 
import numpy as np
import pickle
import queue

import yaml
from dual_pathway_model.functions import *
from dual_pathway_model.directory_functions import *
from dual_pathway_model.model import build_and_run, NN, params_base
from pathlib import Path
# # load parameters from json file
# params_path = "params.json"
# # Open the file and read the contents
# with open(params_path, "r") as f:
#     parameters = json.load(f)

# directory where robustness.py lives
HERE = Path(__file__).resolve().parent

NOS_SEEDS = 100
time_per_iter = 5.5
state = 5
np.random.seed(state)
seeds = np.random.randint(0, 100000, NOS_SEEDS)
seeds.sort()

parameters = update_params(params_base, 
                               **{
                                #    "params.DAYS": 2, # for quick testing
                               }
                               )

# Time required
time_per_iter = 55 # seconds per seed (for 4 syllables)
total_time = time_per_iter * NOS_SEEDS
print(f"Total time required: {total_time/60:.2f} minutes")

# Hoping 4 syllables only
N_SYLL = parameters['params']['N_SYLL']
terminal_performance = np.zeros((NOS_SEEDS, N_SYLL))

if parameters["params"]["N_SYLL"] != 4:
    raise ValueError("N_SYLL must be 4 for robustness analysis.")

for seed_idx, seed in enumerate(seeds):
    perf = build_and_run(seed, parameters, NN)
    terminal_performance[seed_idx, :] = perf
    print(f"Seed {seed} -> {perf}")

results_dir = HERE / "results" / f"multi_syllable"
results_dir.mkdir(parents=True, exist_ok=True)
np.save(results_dir / "terminal_performance.npy", terminal_performance)
with open(results_dir / "meta.yaml", "w") as f:
    yaml.safe_dump(
            {
                "parameter": "multi_syllable",
                "section": "null",
                "seeds": [int(s) for s in seeds],      # convert to Python int
                "shape": terminal_performance.shape
            },
            f
        )
    print(f"Results saved to {results_dir}")
print(terminal_performance[:5, :])  # print first 5 for brevity
print(terminal_performance.shape)



# # if file exists, remove it 
# if os.path.exists("multi_syll_robust.npy"):
#     os.remove("multi_syll_robust.npy")

# def run_mutli_syll_robust(state, nos_seeds, parameters, NN):
#     np.random.seed(state)
#     N_SYLL = parameters['params']['N_SYLL']
#     if N_SYLL == 1:
#         raise Warning("Only one syllable, are you sure?")
#     LANDSCAPE = parameters['params']['LANDSCAPE']
#     # if LANDSCAPE == 0:
#     #     raise Warning("LANDSCAPE is 0, are you sure?")  
#     time_per_iter = 7
#     total_time = time_per_iter * nos_seeds * N_SYLL
#     print(f"Total time: {total_time/60} minutes")
#     total_returns = np.zeros((nos_seeds, 2, 4))
#     seeds = np.random.randint(0, 1000000, nos_seeds)
#     for i, seed in enumerate(seeds):
#         env = Environment(seed, parameters, NN)
#         env.run(parameters, True)
#         final_before_cut = env.rewards[59, -1, :]
#         final_after_cut = env.rewards[60, -1, :]
#         total_returns[i, 0, :] = final_before_cut
#         total_returns[i, 1, :] = final_after_cut
#         print(f"Seed: {seed} with final_before_cut: {final_before_cut} and final_after_cut: {final_after_cut}")
#         print(f"Time remaining: {np.round((total_time - (i+1) * time_per_iter*N_SYLL) / 60, 2)} minutes")
#     return total_returns

# nos_seeds = 100
# state = 5
# total_returns = run_mutli_syll_robust(state, nos_seeds, parameters, NN)

# np.save("multi_syll_robust.npy", total_returns)