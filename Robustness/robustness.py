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

NOS_SEEDS = 2
time_per_iter = 5.5
state = 5
np.random.seed(state)
seeds = np.random.randint(0, 100000, NOS_SEEDS)
seeds.sort()

for param_name, param_info in robustness_cfg.items():
    section = param_info["section"]
    values = param_info["values"]

    print(f"\nRunning robustness for {section}.{param_name}")

    terminal_performance = np.zeros((NOS_SEEDS, len(values)))

    for val_idx, val in enumerate(values):
        print(f" -- {param_name} = {val}")

        parameters = update_params(
            params_base,
            **{
                f"{section}.{param_name}": val,
                "params.N_SYLL": 1
            }
        )

        if parameters["params"]["N_SYLL"] != 1:
            raise ValueError("N_SYLL must be 1 for robustness analysis.")

        for seed_idx, seed in enumerate(seeds):
            perf = build_and_run(seed, parameters, NN)
            terminal_performance[seed_idx, val_idx] = perf
            print(f"    Seed {seed} -> {perf}")

    results_dir = HERE / "results" / f"{section}_{param_name}"
    results_dir.mkdir(parents=True, exist_ok=True)

    np.save(results_dir / "terminal_performance.npy", terminal_performance)

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
