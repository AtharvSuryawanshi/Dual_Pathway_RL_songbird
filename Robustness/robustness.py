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

ROBUSTNESS_CONFIG = HERE / "robustness_params.yaml"

with open(ROBUSTNESS_CONFIG, "r") as f:
    robustness_cfg = yaml.safe_load(f)

print(f"Robustness parameters loaded from {ROBUSTNESS_CONFIG}")

NOS_SEEDS = 100
time_per_iter = 6.5
state = 5
np.random.seed(state)
seeds = np.random.randint(0, 100000, NOS_SEEDS)
seeds.sort()

# TIME ESTIMATE
total_iterations = sum(len(param_info["values"]) for param_info in robustness_cfg.values()) * NOS_SEEDS
total_time_hours = (total_iterations * time_per_iter) / 3600
print(f"Estimated total time for robustness analysis: {total_time_hours:.2f} hours")

for param_name, param_info in robustness_cfg.items():
    section = param_info["section"]
    values = param_info["values"]

    # param_type = param_info.get("type", "float")

    # if param_type == "int":
    #     values = [int(v) for v in param_info["values"]]
    # elif param_type == "float":
    #     values = [float(v) for v in param_info["values"]]
    # else:
    #     raise ValueError(f"Unknown type {param_type} for {param_name}")
    
    print(f"\nRunning robustness for {section}.{param_name}")

    terminal_performance = np.zeros((NOS_SEEDS, len(values)))
    if param_name == "N_DISTRACTORS":
        nos_peaks_output = np.zeros((NOS_SEEDS, len(values)))

    for val_idx, val in enumerate(values):
        val = float(val)  # ensure val is a float for YAML serialization
        print(f" -- {param_name} = {val}")

        parameters = update_params(
            params_base,
            **{
                f"{section}.{param_name}": val,
                "params.N_SYLL": 1,
                # "params.DAYS": 1, # for quick testing
            }
        )

        if parameters["params"]["N_SYLL"] != 1:
            raise ValueError("N_SYLL must be 1 for robustness analysis.")

        for seed_idx, seed in enumerate(seeds):
            if param_name == "N_DISTRACTORS":
                perf, nos_peaks = build_and_run(seed, parameters, NN, find_nos_peaks=True)
            else:
                perf = build_and_run(seed, parameters, NN)
            terminal_performance[seed_idx, val_idx] = perf
            if param_name == "N_DISTRACTORS":
                nos_peaks_output[seed_idx, val_idx] = nos_peaks
            print(f"    Seed {seed} -> {perf}")

    results_dir = HERE / "results" / f"{section}_{param_name}"
    results_dir.mkdir(parents=True, exist_ok=True)

    np.save(results_dir / "terminal_performance.npy", terminal_performance)
    if param_name == "N_DISTRACTORS":
        np.save(results_dir / "nos_peaks.npy", nos_peaks_output)

    with open(results_dir / "meta.yaml", "w") as f:
        yaml.safe_dump(
            {
                "parameter": param_name,
                "section": section,
                "values": [float(v) for v in values],  # convert to Python float
                "seeds": [int(s) for s in seeds],      # convert to Python int
                "shape": terminal_performance.shape
            },
            f
        )
    print(f"Results saved to {results_dir}")
