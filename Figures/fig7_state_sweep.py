"""
Figure 7 state sweep script.
Runs the lesion experiment simulation for states 1000–1499 (500 states),
saves one figure per state to Figures/Plots/fig7_state_sweep/.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

# --- paths ---
HERE = Path(__file__).parent          # Figures/
LESION_CONFIG = HERE / "../Lesion_exp/lesion_params.yaml"
OUT_DIR = HERE / "Plots/fig7_state_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- model imports ---
sys.path.insert(0, str(HERE / "../src"))
from dual_pathway_model.model import NN, params_base, build_and_run
from dual_pathway_model.directory_functions import update_params
from dual_pathway_model.functions import running_mean
from dual_pathway_model.plotting_functions import plot_colors

# --- fixed settings (matching notebook) ---
wanted_days = [5, 30, 55]
NOS_SEEDS = 1
time_per_iter = 5.5

with open(LESION_CONFIG, "r") as f:
    lesion_cfg = yaml.safe_load(f)

# -------------------------------------------------------
def run_one_state(state: int):
    np.random.seed(state)
    seeds = np.sort(np.random.randint(0, 100000, NOS_SEEDS))

    outputs_all = []
    for param_name, param_info in lesion_cfg.items():
        section = param_info["section"]
        values = param_info["values"]
        for val in values:
            val = float(val)
            if val not in wanted_days:
                continue
            parameters = update_params(
                params_base,
                **{
                    f"{section}.{param_name}": val,
                    "params.N_SYLL": 1,
                }
            )
            for seed in seeds:
                outputs = build_and_run(
                    seed, parameters, NN,
                    lesion=True, output_reward=True, output_action=True
                )
                outputs_all.append(outputs)

    rewards_array = np.array([o["rewards"] for o in outputs_all])
    actions_array = np.array([o["actions"] for o in outputs_all])
    return rewards_array, actions_array


def plot_rewards_and_actions(rewards, actions, wanted_days, DAYS_TOTAL, TRIALS_TOTAL, smoothing=10):
    point_colors = {"Before": "grey", "After": "skyblue", "End": "dodgerblue"}
    offset = 40
    rewards = rewards.squeeze()
    actions = actions.squeeze()

    fig, axs = plt.subplots(3, len(wanted_days), figsize=(4 * len(wanted_days), 4), sharey="row")
    if len(wanted_days) == 1:
        axs = np.expand_dims(axs, axis=1)

    for i, lesion_day in enumerate(wanted_days):
        param_idx = i

        reward_data = rewards[param_idx].reshape(-1)
        action_data = actions[param_idx].reshape(-1, 2)
        time_axis = np.linspace(0, DAYS_TOTAL, DAYS_TOTAL * TRIALS_TOTAL)

        reward_before_lesion = reward_data[(lesion_day - 1) * TRIALS_TOTAL : lesion_day * TRIALS_TOTAL]
        reward_after_lesion  = reward_data[lesion_day * TRIALS_TOTAL : (lesion_day + 1) * TRIALS_TOTAL]

        smooth_reward = running_mean(reward_data, smoothing)
        smooth_time   = time_axis[: len(smooth_reward)]

        spacing = 100
        axs[0, i].plot(smooth_time + offset, smooth_reward, color="black", lw=1)
        axs[1, i].plot(smooth_time + offset, smooth_reward, color="black", lw=1)
        axs[0, i].axvline(lesion_day + offset, linestyle="--", color="grey", alpha=0.5)
        axs[1, i].axvline(lesion_day + offset, linestyle="--", color="grey", alpha=0.5)

        axs[0, i].scatter(smooth_time[lesion_day * TRIALS_TOTAL - spacing] + offset, reward_before_lesion.mean(), color=point_colors["Before"], s=100, zorder=5)
        axs[0, i].scatter(smooth_time[lesion_day * TRIALS_TOTAL + spacing] + offset, reward_after_lesion.mean(),  color=point_colors["After"],  s=100, zorder=5)
        axs[1, i].scatter(smooth_time[lesion_day * TRIALS_TOTAL - spacing] + offset, reward_before_lesion.mean(), color=point_colors["Before"], s=100, zorder=5)
        axs[1, i].scatter(smooth_time[lesion_day * TRIALS_TOTAL + spacing] + offset, reward_after_lesion.mean(),  color=point_colors["After"],  s=100, zorder=5)

        axs[0, i].set_xlim(offset + lesion_day - 2, offset + lesion_day + 2)
        axs[0, i].set_ylim(0, 1); axs[0, i].set_yticks([0, 1])
        axs[0, i].spines[["right", "top"]].set_visible(False)

        axs[1, i].set_ylim(0, 1); axs[1, i].set_yticks([0, 1])
        axs[1, i].set_xticks(offset + np.arange(0, DAYS_TOTAL + 1, 20))
        axs[1, i].spines[["right", "top"]].set_visible(False)

        if i == 0:
            axs[0, i].set_ylabel("Reward", fontsize=12)
            axs[1, i].set_ylabel("Reward", fontsize=12)

        motor_var = np.std(actions[param_idx], axis=1)
        mean_motor_var = np.mean(motor_var, axis=1)

        axs[2, i].plot(np.arange(1, DAYS_TOTAL + 1) + offset, mean_motor_var, color="black")
        axs[2, i].axvline(lesion_day + offset, linestyle="--", color="grey", alpha=0.5)
        axs[2, i].scatter(lesion_day - 1 + offset, mean_motor_var[lesion_day - 1], color=point_colors["Before"], s=100, zorder=5)
        axs[2, i].scatter(lesion_day     + offset, mean_motor_var[lesion_day],     color=point_colors["After"],  s=100, zorder=5)
        axs[2, i].scatter(DAYS_TOTAL - 1 + offset, mean_motor_var[DAYS_TOTAL - 1], color=point_colors["End"],    s=100, zorder=5)
        axs[2, i].set_xlim(offset + lesion_day - 2, offset + lesion_day + 2)
        axs[2, i].set_ylim(0, 1); axs[2, i].set_yticks([0, 1])
        axs[2, i].set_xlabel(lesion_day + offset, fontsize=12)
        axs[2, i].spines[["right", "top"]].set_visible(False)
        if i == 0:
            axs[2, i].set_ylabel("Motor variability", fontsize=12)

    plt.tight_layout()
    return fig


# -------------------------------------------------------
STATES = range(1000, 1500)
total = len(STATES)

for idx, state in enumerate(STATES):
    out_path = OUT_DIR / f"state_{state:04d}.png"
    if out_path.exists():
        print(f"[{idx+1}/{total}] state={state} already exists, skipping.")
        continue

    print(f"[{idx+1}/{total}] state={state} ...", flush=True)
    try:
        rewards_array, actions_array = run_one_state(state)

        DAYS   = rewards_array.shape[1]
        TRIALS = rewards_array.shape[2]

        fig = plot_rewards_and_actions(rewards_array, actions_array, wanted_days, DAYS, TRIALS, smoothing=10)
        fig.suptitle(f"state = {state}", fontsize=10, y=1.01)
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"    saved -> {out_path.name}", flush=True)
    except Exception as e:
        print(f"    ERROR for state={state}: {e}", flush=True)

print("Done.")
