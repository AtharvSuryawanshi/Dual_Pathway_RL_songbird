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

LESION_CONFIG = HERE / "lesion_params.yaml"

with open(LESION_CONFIG, "r") as f:
    lesion_cfg = yaml.safe_load(f)

print(f"Lesion parameters loaded from {LESION_CONFIG}")

NOS_SEEDS = 3
time_per_iter = 5.5
state = 5
np.random.seed(state)
seeds = np.random.randint(0, 100000, NOS_SEEDS)
seeds.sort()

# TIME ESTIMATE
total_iterations = sum(len(param_info["values"]) for param_info in lesion_cfg.values()) * NOS_SEEDS
total_time_hours = (total_iterations * time_per_iter) / 3600
print(f"Estimated total time for robustness analysis: {total_time_hours:.2f} hours")

for param_name, param_info in lesion_cfg.items():
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

    terminal_performance = np.zeros((NOS_SEEDS, len(values), 3))
    terminal_motor_var = np.zeros((NOS_SEEDS, len(values), 3))

    for val_idx, val in enumerate(values):
        val = float(val)  # ensure val is a float for YAML serialization
        print(f" -- {param_name} = {val}")

        parameters = update_params(
            params_base,
            **{
                f"{section}.{param_name}": val,
                "params.N_SYLL": 1,
                "params.DAYS": 62, # for quick testing
            }
        )

        if parameters["params"]["N_SYLL"] != 1:
            raise ValueError("N_SYLL must be 1 for robustness analysis.")

        for seed_idx, seed in enumerate(seeds):
            if param_name == "BG_INTACT_DAYS":
                terminal_perf, before_lesion, after_lesion, motor_var_terminal, motor_var_before, motor_var_after = build_and_run(seed, parameters, NN, lesion_bg = True, motor_variability = True)
                terminal_performance[seed_idx, val_idx, :] = terminal_perf, before_lesion, after_lesion
                print(motor_var_terminal, motor_var_before, motor_var_after)
                terminal_motor_var[seed_idx, val_idx, :] = motor_var_terminal, motor_var_before, motor_var_after
                print(f"    Seed {seed} -> {terminal_perf}, {before_lesion}, {after_lesion}")
            elif param_name == "RA_INTACT_DAYS":
                terminal_perf, before_lesion, after_lesion, motor_var_terminal, motor_var_before, motor_var_after = build_and_run(seed, parameters, NN, lesion_ra = True, motor_variability = True)
                terminal_performance[seed_idx, val_idx, :] = terminal_perf, before_lesion, after_lesion
                print(motor_var_terminal, motor_var_before, motor_var_after)
                terminal_motor_var[seed_idx, val_idx, :] = motor_var_terminal, motor_var_before, motor_var_after
                print(f"    Seed {seed} -> {terminal_perf}, {before_lesion}, {after_lesion}")
            else:
                raise ValueError(f"Unknown lesion parameter {param_name}")

    results_dir = HERE / "results" / f"{section}_{param_name}"
    results_dir.mkdir(parents=True, exist_ok=True)

    np.save(results_dir / "terminal_performance.npy", terminal_performance)
    np.save(results_dir / "terminal_motor_var.npy", terminal_motor_var)

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
