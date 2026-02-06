# Import important libraries
import matplotlib.pyplot as plt # DUH
import numpy as np
import seaborn as sns
from pathlib import Path
import yaml
from matplotlib.colors import LinearSegmentedColormap
from dual_pathway_model.functions import *

# Parameters 
# Get path relative to this file
config_path = Path(__file__).parent / "plotting_colors.yaml"

with open(config_path, "r") as f:
    plot_colors = yaml.safe_load(f)
    print("Plotting colors safely loaded")


def plot_results_violin(returns, params, 
                        plot_colors = plot_colors, 
                        big_xlabel=None, xticklabels=None,
                        print_success_rate=True
                        ):
    returns = np.asarray(returns)
    sorted_params = list(params)
    n_values = len(sorted_params)
    above_threshold2 = np.zeros(n_values)
    data, labels_list = [], []

    for i in range(n_values):
        col = returns[:, i]
        above_threshold2[i] = np.mean(col > 0.7)
        data.extend(col)
        labels_list.extend([sorted_params[i]] * len(col))

    # --- Figure & layout ---
    fig = plt.figure(
        figsize=(1.5 * n_values + 2, 7),
        constrained_layout=True
    )
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[6, 1.2],
        height_ratios=[2, 5],
        wspace=0.25,
        hspace=0.15
    )
    # =======================
    # Violin + strip plot
    # =======================
    ax1 = fig.add_subplot(gs[:, 0])

    sns.violinplot(
        x=labels_list,
        y=data,
        inner=None,
        color=plot_colors['violin_plot_colors']['color_violin'],
        linewidth=0,
        cut=0,
        bw_adjust=0.5,
        width=0.5,
        ax=ax1
    )

    # Clip violins to right half
    for c in ax1.collections:
        if hasattr(c, "get_paths"):
            for path in c.get_paths():
                verts = path.vertices
                x_center = np.median(verts[:, 0])
                verts[:, 0] = np.maximum(verts[:, 0], x_center)

    strip = sns.stripplot(
        x=labels_list,
        y=data,
        size=3,
        color=plot_colors['violin_plot_colors']['color_strip'],
        alpha=0.9,
        jitter=0.05,
        ax=ax1
    )

    # Shift strip slightly left
    for coll in strip.collections:
        offsets = coll.get_offsets()
        offsets[:, 0] -= 0.1
        coll.set_offsets(offsets)

    ax1.axhline(
        0.7, 0.05, 0.95,
        color='grey',
        linestyle='--',
        linewidth=1,
        label='Global maxima threshold'
    )

    ax1.set_ylim(0, 1)
    ax1.set_yticks([0, 0.7, 1])
    ax1.set_yticklabels(['0', '0.7', '1'], fontsize=12)
    ax1.set_ylabel('Terminal Performance', fontsize=16)

    if big_xlabel is not None:
        ax1.set_xlabel(big_xlabel, fontsize=16)

    # 🔒 FIX: ticks BEFORE labels
    ax1.set_xticks(range(n_values))
    ax1.set_xticklabels(
        xticklabels if xticklabels is not None else sorted_params,
        fontsize=12
    )

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(loc='lower right', fontsize=11, frameon=False)

    # =======================
    # Success-rate bar plot
    # =======================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(
        range(n_values),
        above_threshold2 * 100,
        color= plot_colors['violin_plot_colors']['color_bar'],
    )
    # plot above_threshold2 values on top of bars
    if print_success_rate:
        for i, val in enumerate(above_threshold2 * 100):
            ax2.text(
                i, val + 2, f"{val:.1f}",
                ha='center', va='bottom', fontsize=10, rotation=90

            )

    ax2.set_ylim(0, 100)
    ax2.set_ylabel('Success Rate (%)', fontsize=12)

    ax2.set_xticks(range(n_values))
    ax2.set_xticklabels(
        xticklabels if xticklabels is not None else sorted_params,
        rotation=90,
        ha='right',
        va='center',
        fontsize=10,
        rotation_mode="anchor"
    )

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.show()


# LANDSCAPE PLOT
def plot_lansdcape_only(obj, syll, plot_colors = plot_colors):
    fig, axs = plt.subplots(figsize=(9, 9))
    cmap = LinearSegmentedColormap.from_list('change_this', ['white', 'white']) # 'Greys' color_contour_bckg #'Purples' #LinearSegmentedColormap.from_list('white_to_black', ['white', 'rebeccapurple'])
    levels_ = 50
    ##### Artificial Landscapes #####
    def plot_artificial():
        limit = obj.limit
        x, y = np.linspace(-limit, limit, 50), np.linspace(-limit, limit, 50)
        X, Y = np.meshgrid(x, y)
        Z = obj.get_reward([X, Y], syll)
        contour = axs.contourf(X, Y, Z, levels=levels_, cmap=cmap)
        if if_contour:
            axs.contour(X, Y, Z, levels=15, colors='k', linewidths=1, alpha=0.9)
        # cbar = fig.colorbar(contour, ax=axs)
        # cbar.set_label('Performance Metric (R)', fontsize=20, rotation = 270)
        # cbar.ax.tick_params(labelsize=18)
        # cbar.ax.set_yticks([0,1])

        axs.set_xticks([-limit, 0, limit], [-1, 0, 1])
        axs.set_yticks([-limit, 0, limit], [-1, 0, 1])
    ##### Syrinx Landscapes #####
    def plot_syrinx():
        if obj.N_SYLL > 4:
            raise ValueError('Only 4 syllables are available in the syrinx landscape')
        obj.syrinx_contours = []
        obj.syrinx_targets = []
        for j in range(obj.N_SYLL):
            base = np.load(f"contours/Syll{j+1}.npy")
            Z, target = make_contour(base)
            obj.syrinx_contours.append(Z)
            obj.syrinx_targets.append(target)
        obj.centers = np.array(obj.syrinx_targets)
        obj.syrinx_contours = np.array(obj.syrinx_contours)
        Z = obj.syrinx_contours[syll]
        target_pos = obj.syrinx_targets[syll]
        if if_contour:
            axs.contour(Z.T, levels=15, extent=[-1, 1, -1, 1], colors='k', linewidths=1, alpha=1)
        cs = axs.contourf(Z.T, cmap=cmap, extent=[-1, 1, -1, 1], levels=levels_)
        # cbar = fig.colorbar(cs, ax = axs)
        # cbar.set_label('Performance Metric (R)', fontsize=20, rotation = 270)
        # cbar.ax.tick_params(labelsize=18)
        # cbar.ax.set_yticks([0,1])
        axs.set_xticks([-1, 1], [0, 1])
        axs.set_yticks([-1,1], [0, 0.2])
        # axs.scatter(target_pos[1], target_pos[0], s=100, c='green', marker='x', label='Target')
    if obj.LANDSCAPE == 0:
        print("Plotting artificial landscape")
        plot_artificial()
    else:
        print("Plotting syrinx landscape")
        plot_syrinx()
    # if not force_landscape:
    #     if obj.LANDSCAPE == 0:
    #         print("No force artificial landscape")
    #         plot_artificial()
    #     else:
    #         plot_syrinx()
    # else:
    #     if landscape == 0:
    #         plot_artificial()
    #     else:
    #         plot_syrinx()

    # axs.set_ylabel(r'$P_{\alpha}$', fontsize=22)
    # axs.set_xlabel(r'$P_{\beta}$', fontsize=22)
    axs.set_ylabel(r'$P$', fontsize=22)
    axs.set_xlabel(r'$T$', fontsize=22)
    axs.tick_params(labelsize=16)
    axs.legend()
    plt.tight_layout()
    plt.show()

