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

pathway_colors = plot_colors['colors_pathway']
sns_cmap = pathway_colors['palette']
color_motor = pathway_colors['motor']


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
def plot_artificial(obj, syll, axs, levels_, cmap, if_contour):
    limit = obj.limit
    x, y = np.linspace(-limit, limit, 50), np.linspace(-limit, limit, 50)
    X, Y = np.meshgrid(x, y)
    Z = obj.get_reward([X, Y], syll)
    contour = axs.contourf(X, Y, Z, levels=levels_, cmap=cmap)
    if if_contour:
        axs.contour(X, Y, Z, levels=15, colors='k', linewidths=1, alpha=0.9)
    # cbar = fig.colorbar(contour, ax=axs)
    # cbar.set_label('Performance Metric (R)', fontsize=20, rotation=270)
    # cbar.ax.tick_params(labelsize=18)
    # cbar.ax.set_yticks([0, 1])
    axs.set_xticks([-limit, 0, limit], [-1, 0, 1])
    axs.set_yticks([-limit, 0, limit], [-1, 0, 1])

def plot_syrinx(obj, syll, axs, levels_, cmap, if_contour):
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
        axs.contour(Z.T, levels=10, extent=[-1, 1, -1, 1], colors='k', linewidths=1, alpha=1)
    cs = axs.contourf(Z.T, cmap=cmap, extent=[-1, 1, -1, 1], levels=levels_)
    # cbar = fig.colorbar(cs, ax=axs)
    # cbar.set_label('Performance Metric (R)', fontsize=20, rotation=270)
    # cbar.ax.tick_params(labelsize=18)
    # cbar.ax.set_yticks([0, 1])
    axs.set_xticks([-1, 1], [0, 1])
    axs.set_yticks([-1, 1], [0, 0.2])
    # axs.scatter(target_pos[1], target_pos[0], s=100, c='green', marker='x', label='Target')


def plot_lansdcape_only(obj, syll, plot_colors = plot_colors):
    fig, axs = plt.subplots(figsize=(9, 9))
    cmap = LinearSegmentedColormap.from_list('change_this', ['white', 'white']) # 'Greys' color_contour_bckg #'Purples' #LinearSegmentedColormap.from_list('white_to_black', ['white', 'rebeccapurple'])
    levels_ = 50
    ##### Artificial Landscapes #####
    # def plot_artificial():
    #     limit = obj.limit
    #     x, y = np.linspace(-limit, limit, 50), np.linspace(-limit, limit, 50)
    #     X, Y = np.meshgrid(x, y)
    #     Z = obj.get_reward([X, Y], syll)
    #     contour = axs.contourf(X, Y, Z, levels=levels_, cmap=cmap)
    #     if if_contour:
    #         axs.contour(X, Y, Z, levels=15, colors='k', linewidths=1, alpha=0.9)
    #     # cbar = fig.colorbar(contour, ax=axs)
    #     # cbar.set_label('Performance Metric (R)', fontsize=20, rotation = 270)
    #     # cbar.ax.tick_params(labelsize=18)
    #     # cbar.ax.set_yticks([0,1])

    #     axs.set_xticks([-limit, 0, limit], [-1, 0, 1])
    #     axs.set_yticks([-limit, 0, limit], [-1, 0, 1])
    # ##### Syrinx Landscapes #####
    # def plot_syrinx():
    #     if obj.N_SYLL > 4:
    #         raise ValueError('Only 4 syllables are available in the syrinx landscape')
    #     obj.syrinx_contours = []
    #     obj.syrinx_targets = []
    #     for j in range(obj.N_SYLL):
    #         base = np.load(f"contours/Syll{j+1}.npy")
    #         Z, target = make_contour(base)
    #         obj.syrinx_contours.append(Z)
    #         obj.syrinx_targets.append(target)
    #     obj.centers = np.array(obj.syrinx_targets)
    #     obj.syrinx_contours = np.array(obj.syrinx_contours)
    #     Z = obj.syrinx_contours[syll]
    #     target_pos = obj.syrinx_targets[syll]
    #     if if_contour:
    #         axs.contour(Z.T, levels=15, extent=[-1, 1, -1, 1], colors='k', linewidths=1, alpha=1)
    #     cs = axs.contourf(Z.T, cmap=cmap, extent=[-1, 1, -1, 1], levels=levels_)
    #     # cbar = fig.colorbar(cs, ax = axs)
    #     # cbar.set_label('Performance Metric (R)', fontsize=20, rotation = 270)
    #     # cbar.ax.tick_params(labelsize=18)
    #     # cbar.ax.set_yticks([0,1])
    #     axs.set_xticks([-1, 1], [0, 1])
    #     axs.set_yticks([-1,1], [0, 0.2])
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

def plot_scatter_traj(obj, syll, day_i, day_f, every_nth_point,
                      plot_smooth_traj = False, N_i=5, N_f=1, steepness=20, figsize=(10,10), plot_colors = plot_colors):
    fig, axs = plt.subplots(figsize=figsize)
    cmap = color_contour_bckg # Match the colormap style from plot_landscape
    levels_ = 50
    TRIALS = obj.TRIALS

    # Plot background landscape
    if obj.LANDSCAPE == 0:
        print("Plotting artificial landscape")
        plot_artificial(obj, syll, axs, levels_, cmap, if_contour=True)
    else:
        print("Plotting syrinx landscape")
        plot_syrinx(obj, syll, axs, levels_, cmap, if_contour=True)
    
    # Plot agent trajectory
    x_traj, y_traj = zip(*obj.actions[:, :, syll, :].reshape(-1, 2))
    # axs.plot(
    #     x_traj[day_i * TRIALS: day_f * TRIALS][::every_nth_point],
    #     y_traj[day_i * TRIALS: day_f * TRIALS][::every_nth_point], color = color_motor, label='Agent Trajectory', alpha=.5, linewidth=0.0, marker='.', markersize=1
    # )
    if plot_smooth_traj:
        xtraj_smooth = running_mean_dynamic(np.array(x_traj), N_i=N_i, N_f=N_f, steepness=steepness)
        ytraj_smooth = running_mean_dynamic(np.array(y_traj), N_i=N_i, N_f=N_f, steepness=steepness)
        axs.plot(
            xtraj_smooth[day_i * TRIALS: day_f * TRIALS], #[::every_nth_point],
            ytraj_smooth[day_i * TRIALS: day_f * TRIALS],#[::every_nth_point],
            color = color_motor, label='Motor Trajectory', alpha=.5, linewidth=0.5, marker='.', markersize=1, ls='-'
        )
    
    axs.plot(
        x_traj[day_i * TRIALS: day_f * TRIALS][::every_nth_point],
        y_traj[day_i * TRIALS: day_f * TRIALS][::every_nth_point], color = 'white', label='Agent Trajectory', alpha=1, linewidth=0.0, marker='.', markersize=3, zorder=3
    )
    axs.plot(
        x_traj[day_i * TRIALS: day_f * TRIALS][::every_nth_point],
        y_traj[day_i * TRIALS: day_f * TRIALS][::every_nth_point], color = color_motor, label='Agent Trajectory', alpha=1, linewidth=0.0, marker='.', markersize=2, zorder=3
    )
    
    axs.scatter(x_traj[0], y_traj[0],
                s=150, c='black',
                marker='s', zorder=5, label='Starting Point')
    axs.scatter(x_traj[-1], y_traj[-1],
                s=150, c='white',
                marker='X', zorder=5, label='Ending Point')
    axs.scatter(x_traj[0], y_traj[0],
                s=50, c='white',
                marker='s', zorder=6, label='Starting Point')
    axs.scatter(x_traj[-1], y_traj[-1],
                s=50, c='black',
                marker='x', zorder=6, label='Ending Point')

    # Labels
    axs.set_ylabel(r'$P$', fontsize=22)
    axs.set_xlabel(r'$T$', fontsize=22)
    # axs.set_ylabel(r'$P_{\alpha}$ (Pressure)', fontsize=22)
    # axs.set_xlabel(r'$P_{\beta}$ (Tension)', fontsize=22)
    axs.tick_params(labelsize=16)
    # axs.legend()
    plt.tight_layout()
    # plt.savefig(figures_path+'contour_syll'+str(syll+1)+'.png')
    plt.show()



###### PLOT MOTOR OUTPUTS ######
def plot_output(obj, syll):
    figure, (ax1, ax2) = plt.subplots(2,1)
    N_SYLL = obj.N_SYLL # To plot only one syllable at a time, set N_SYLL to 1 and plot the first syllable (syll=0)
    N_DAILY_MOTIFS = obj.TRIALS
    DAYS = obj.DAYS
    sk = N_SYLL
    LIMIT = 1.5

    # Display x axis in days
    x = np.arange(N_DAILY_MOTIFS)
    x = x/(N_DAILY_MOTIFS * N_SYLL)

    x_bg_traj, y_bg_traj = zip(*obj.actions_bg[:, :, syll, :].reshape(-1, 2))
    ra_actions = obj.actions - obj.actions_bg
    x_ra_traj, y_ra_traj = zip(*ra_actions[:, :, syll, :].reshape(-1, 2))
    x_bg_traj = np.array(x_bg_traj)
    y_bg_traj = np.array(y_bg_traj)
    x_ra_traj = np.array(x_ra_traj)
    y_ra_traj = np.array(y_ra_traj)


    # Plot running average of cortical output (brown), BG output (grey) and total output (black)
    # Data
    # for syll in range(N_SYLL):
        # motor outputs
    ax1.plot(running_mean(obj.actions[:,:,syll,0].reshape(DAYS*N_DAILY_MOTIFS), 1), color=color_motor, lw=1, alpha=.9)
    ax2.plot(running_mean(obj.actions[:,:,syll,1].reshape(DAYS*N_DAILY_MOTIFS), 1), color=color_motor, lw=1, alpha=.9)

    ## RA contribution 
    # ax1.plot(running_mean(x_ra_traj.reshape(DAYS*N_DAILY_MOTIFS), 1), color="brown", lw=1, alpha=.9, label='RA contribution')
    # ax2.plot(running_mean(y_ra_traj.reshape(DAYS*N_DAILY_MOTIFS), 1), color="brown", lw=1, alpha=.9)
    # #     # BG contribution
    # ax1.plot(running_mean(x_bg_traj.reshape(DAYS*N_DAILY_MOTIFS), 10), color="grey", lw=1, alpha=.9, label='BG contribution')
    # ax2.plot(running_mean(y_bg_traj.reshape(DAYS*N_DAILY_MOTIFS), 10), color="grey", lw=1, alpha=.9)
    # for syll in range(N_SYLL - 1):
    #     # target
    #     ax1.plot(obj.centers[syll, 0]*np.ones(N_DAILY_MOTIFS*DAYS),  color='red', linestyle='--', linewidth=1)
    #     ax2.plot(obj.centers[syll, 1]*np.ones(N_DAILY_MOTIFS*DAYS),  color='red', linestyle='--', linewidth=1)
    # if N_SYLL > 1:
    #     ax1.plot(obj.centers[syll + 1, 0]*np.ones(N_DAILY_MOTIFS*DAYS),  color='red', linestyle='--', linewidth=1)
    #     ax2.plot(obj.centers[syll + 1, 1]*np.ones(N_DAILY_MOTIFS*DAYS),  color='red', linestyle='--', linewidth=1, label = 'Target')
    # elif N_SYLL == 1:
    ax1.plot(obj.centers[syll, 1]*np.ones(N_DAILY_MOTIFS*DAYS),  color='grey', linestyle='--', linewidth=1, label = 'Target')
    ax2.plot(obj.centers[syll, 0]*np.ones(N_DAILY_MOTIFS*DAYS),  color='grey', linestyle='--', linewidth=1)

    # Axis beauty
    # ax1.axvline(x=N_DAYS_INTACT * N_DAILY_MOTIFS, linestyle='--', color='grey', lw=1)
    # ax2.axvline(x=N_DAYS_INTACT * N_DAILY_MOTIFS, linestyle='--', color='grey', lw=1)
    # ax1.axhline(y=0, linestyle='--', color='black', alpha=0.1)
#         ax1.axhline(y=obj.targetpos[0], linestyle='--', color='black', label='Global optimum')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.get_xaxis().set_ticks([])
    ax1.set_ylim(-LIMIT, LIMIT)
    ax1.tick_params(labelsize=15)
    ax1.set_yticks([-LIMIT, 0, LIMIT], [-1, 0, 1] )
    ax1.set_xlim(-N_DAILY_MOTIFS, N_DAILY_MOTIFS*(DAYS))
    
    ax2.set_xlabel('DPH', fontsize=20)
    ax2.set_yticks([-LIMIT, 0, LIMIT], [-1, 0, 1]   )
    ax2.set_ylim(-LIMIT, LIMIT)
    ax2.tick_params(labelsize=15)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlim(-N_DAILY_MOTIFS, N_DAILY_MOTIFS*DAYS)
    # ax2.spines['bottom'].set_bounds(0, obj.n_days+obj.n_lesioned_days)
    
    ax2.set_ylabel(r'$P$', fontsize=20)
    ax1.set_ylabel(r'$T$', fontsize=20)
    plt.legend(frameon=False, loc='center right', fontsize=12, bbox_to_anchor=(1.03,1))
    plt.xticks(range(0, N_DAILY_MOTIFS*(DAYS+1), 20*N_DAILY_MOTIFS), np.arange(40, DAYS+1+40, 20))
    ax1.legend()
    plt.tight_layout()
    # plt.savefig(figures_path+'motor_output'+'.png')
    plt.show()