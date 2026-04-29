import os 
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import pickle
import queue
from scipy.integrate import solve_ivp
from dual_pathway_model.functions import *
from dual_pathway_model.directory_functions import *
from dual_pathway_model.model import build_and_run, NN, params_base

# directory where robustness.py lives
HERE = Path(__file__).resolve().parent

# # ROBUSTNESS_CONFIG = HERE / "robustness_params.yaml"

# with open(ROBUSTNESS_CONFIG, "r") as f:
#     robustness_cfg = yaml.safe_load(f)

# print(f"Robustness parameters loaded from {ROBUSTNESS_CONFIG}")

NOS_SEEDS = 100
time_per_iter = 6.5
state = 5
np.random.seed(state)
seeds = np.random.randint(0, 100000, NOS_SEEDS)
seeds.sort()

# TIME ESTIMATE
total_iterations = NOS_SEEDS
total_time_hours = (total_iterations * time_per_iter) / 3600
print(f"Estimated total time for robustness analysis: {total_time_hours:.2f} hours")

parameters = update_params(
    params_base,
    **{
        # f"{section}.{param_name}": val,
        "params.N_SYLL": 1,
        # "params.DAYS": 5, # for quick testing
    }
)

# for param_name, param_info in robustness_cfg.items():
#     section = param_info["section"]
#     values = param_info["values"]

#     # param_type = param_info.get("type", "float")

#     # if param_type == "int":
#     #     values = [int(v) for v in param_info["values"]]
#     # elif param_type == "float":
#     #     values = [float(v) for v in param_info["values"]]

#     # else:
#     #     raise ValueError(f"Unknown type {param_type} for {param_name}")
    
#     print(f"\nRunning robustness for {section}.{param_name}")

terminal_performance = np.zeros((NOS_SEEDS))
motor_displacement_array = np.zeros((NOS_SEEDS, parameters["params"]["DAYS"]-1, parameters["params"]["N_SYLL"], parameters["const"]["MC_SIZE"]))
reward_displacement_array = np.zeros((NOS_SEEDS, parameters["params"]["DAYS"]-1, parameters["params"]["N_SYLL"]))
    # if param_name == "N_DISTRACTORS":
    #     nos_peaks_output = np.zeros((NOS_SEEDS, len(values)))

    # for val_idx, val in enumerate(values):
    #     val = float(val)  # ensure val is a float for YAML serialization
    #     print(f" -- {param_name} = {val}")



if parameters["params"]["N_SYLL"] != 1:
    raise ValueError("N_SYLL must be 1 for robustness analysis.")

for seed_idx, seed in enumerate(seeds):
    output = build_and_run(seed, parameters, NN, terminal_performance=True, motor_displacement=True, reward_displacement=True)
    # print(output.keys())
    # print(output)
    perf = output['terminal_performance_reward'][0]
    # print(perf.shape)
    motor_displacement = output['motor_displacement']
    reward_displacement = output['reward_displacement']
    terminal_performance[seed_idx] = perf
    motor_displacement_array[seed_idx] = motor_displacement
    reward_displacement_array[seed_idx] = reward_displacement
    # print(f"Seed {seed} -> Terminal performance: {perf}")
    # print(f"Motor displacement shape: {motor_displacement}")
    # print(f"Motor displacement shape: {motor_displacement.shape}")
    # if param_name == "N_DISTRACTORS":
    #     nos_peaks_output[seed_idx, val_idx] = nos_peaks
    # print(f"    Seed {seed} -> {perf}")

results_dir = HERE / "results" / f"param_stat_test_2"
results_dir.mkdir(parents=True, exist_ok=True)

np.save(results_dir / "terminal_performance.npy", terminal_performance)
np.save(results_dir / "motor_displacement.npy", motor_displacement_array)
np.save(results_dir / "reward_displacement.npy", reward_displacement_array)
# Print saved for verification
print(f"Terminal performance saved: {terminal_performance.shape}")
print(f"Motor displacement shape saved: {motor_displacement_array.shape}") 
print(f"Reward displacement shape saved: {reward_displacement_array.shape}")

# if param_name == "N_DISTRACTORS":
#     np.save(results_dir / "nos_peaks.npy", nos_peaks_output)

with open(results_dir / "meta.yaml", "w") as f:
    yaml.safe_dump(
        {
            "parameter": "stat_test_2",
            # "section": "params",
            # "values": [float(v) for v in values],  # convert to Python float
            "seeds": [int(s) for s in seeds],      # convert to Python int
            "shape": terminal_performance.shape,
            "motor_displacement_shape": motor_displacement_array.shape,
            "reward_displacement_shape": reward_displacement_array.shape
        },
        f
    )
print(f"Results saved to {results_dir}")
