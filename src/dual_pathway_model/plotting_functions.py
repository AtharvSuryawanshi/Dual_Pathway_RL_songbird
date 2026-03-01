# Import important libraries
import matplotlib.pyplot as plt # DUH
import numpy as np
import seaborn as sns
from pathlib import Path
import yaml
from matplotlib.colors import LinearSegmentedColormap
from dual_pathway_model.functions import *
from matplotlib.collections import LineCollection
import os

print('Testing')

# Parameters 
# Get path relative to this file
config_path = Path(__file__).parent / "plotting_colors.yaml"

with open(config_path, "r") as f:
    plot_colors = yaml.safe_load(f)
    print("Plotting colors safely loaded")

pathway_colors = plot_colors['colors_pathway']
# sns_cmap = pathway_colors['palette']
color_motor = pathway_colors['motor']


def plot_results_violin(returns, params, 
                        plot_colors = plot_colors, 
                        big_xlabel=None, xticklabels=None,
                        xticklabel_rotation=0,
                        print_success_rate=True,
                        height_ratio=[2.4, 5]
                        ):
    returns = np.asarray(returns)
    sorted_params = list(params)
    n_values = len(sorted_params)
    above_threshold2 = np.zeros(n_values)
    std_vals = np.zeros(n_values)
    mean_vals = np.zeros(n_values)
    data, labels_list = [], []

    for i in range(n_values):
        col = returns[:, i]
        above_threshold2[i] = np.mean(col > 0.7)
        std_vals[i] = np.std(col)
        mean_vals[i] = np.mean(col)
        data.extend(col)
        labels_list.extend([sorted_params[i]] * len(col))
    print("Above threshold success rates:", above_threshold2)
    print("Standard deviations:", std_vals*100)
    print("Mean values:", mean_vals*100)


    # --- Figure & layout ---
    fig = plt.figure(
        figsize=(1.5 * n_values + 2, 7),
        constrained_layout=True
    )
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[6, 2],
        height_ratios=height_ratio,
        wspace=0.15,
        hspace=0
    )
    # =======================
    # Violin + strip plot
    # =======================
    ax1 = fig.add_subplot(gs[:, 0])

    ax1.axhspan(.7, 1.2, alpha=.25, color='grey')


    sns.violinplot(
        x=labels_list,
        y=data,
        inner=None,
        color= 'grey', #plot_colors['violin_plot_colors']['color_violin'],
        linewidth=0,
        cut=0,
        bw_adjust=.5,
        width=1,
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
        jitter=0.1,
        ax=ax1
    )

    # Shift strip slightly left
    for coll in strip.collections:
        offsets = coll.get_offsets()
        offsets[:, 0] -= 0.2
        coll.set_offsets(offsets)

    ax1.axhline(
        0.7, 0.02, .98,
        color='dimgray',
        linestyle='--',
        linewidth=2,
        label='Success threshold'
    )


    ax1.set_ylim(0, 1.015)
    ax1.spines['left'].set_bounds(0, 1)
    ax1.set_yticks([0, 0.7, 1])
    ax1.set_yticklabels(['0', '0.7', '1'], fontsize=12)
    ax1.set_ylabel('Terminal Performance', fontsize=16)

    if big_xlabel is not None:
        ax1.set_xlabel(big_xlabel, fontsize=16, labelpad=10)

    # 🔒 FIX: ticks BEFORE labels
    ax1.set_xlim(-.45, n_values-.4)
    ax1.set_xticks(np.arange(n_values)-.05)
    ax1.set_xticklabels(
        xticklabels if xticklabels is not None else sorted_params,
        fontsize=12
    )
    ax1.tick_params(axis='x', which='major', length=0)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(loc='lower right', fontsize=11, facecolor='lightgrey')

    # =======================
    # Success-rate bar plot
    # =======================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(
        range(n_values),
        above_threshold2 * 100,
        alpha=.75,
        color='grey'# plot_colors['violin_plot_colors']['color_bar'],
    )
    # plot above_threshold2 values on top of bars
    if print_success_rate:
        for i, val in enumerate(above_threshold2 * 100):
            ax2.text(
                i, val + 5, int(val), # f"{val:.1f}",
                ha='center', va='bottom', fontsize=10, rotation=0
            )

    ax2.set_ylim(0, 100)
    ax2.set_ylabel('Success Rate (%)', fontsize=12, rotation=270)
    ax2.set_yticks([0, 100])
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')


    ax2.set_xticks(range(n_values))
    ax2.tick_params(axis='x', which='major', length=0)

    ha = 'right' if xticklabel_rotation != 0 else 'center'
    ax2.set_xticklabels(
        xticklabels if xticklabels is not None else sorted_params,
        rotation=xticklabel_rotation,
        ha=ha,
        va='top',
        fontsize=10,
        rotation_mode="anchor"
    )

    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    print("Means")
    # plt.show()


from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
# import lin
import yaml
# color_contour_bckg = LinearSegmentedColormap.from_list('change_this', ['white', 'white']) 
# Parameters 
# Get path relative to this file
# config_path = Path(__file__).parent / "plotting_colors.yaml"
# with open(config_path, "r") as f:
#     plot_colors = yaml.safe_load(f)
#     print("Plotting colors safely loaded")

# pathway_colors = plot_colors['colors_pathway']
# sns_cmap = pathway_colors['palette']
# color_motor = pathway_colors['motor']


# LANDSCAPE PLOT

def plot_artificial(obj, syll, axs, levels_, cmap, if_contour, contour_alpha=1, heatmap=False, colorbar=False):
    
    if not heatmap and colorbar:
        print("Warning: Colorbar is only plotted when heatmap is True. Setting colorbar to False.")
        colorbar = False
    
    limit = obj.limit
    x, y = np.linspace(-limit, limit, 50), np.linspace(-limit, limit, 50)
    X, Y = np.meshgrid(x, y)
    Z = obj.get_reward([X, Y], syll)
    # Z should be normalised for the simulations - max 1 min 0
    if if_contour:
        axs.contour(X, Y, Z, levels=levels_, extent=[-limit, limit, -limit, limit], aspect='equal', colors='k', linewidths=1, alpha=contour_alpha)
    if heatmap:
        cs = axs.contourf(Z, cmap=cmap, extent=[-limit, limit, -limit, limit], aspect='equal', vmin=0, vmax=1, levels=levels_, alpha=contour_alpha)
        if colorbar:
            # Create colorbar on the inset axis (cax)
            cax = axs.inset_axes((1.05, 0, 0.08, 1.0))
            cbar = axs.figure.colorbar(cs, cax=cax)
            cbar.set_label('Performance Metric (R)', fontsize=30, rotation=270, labelpad=20)
            cbar.ax.tick_params(labelsize=20)
            cbar.ax.set_yticks([0, 1])
    axs.set_aspect('equal', adjustable='box')
    axs.set_xticks([-limit, limit], [-1, 1])
    axs.set_yticks([limit], [1])
    axs.set_xlim([-limit, limit])
    axs.set_ylim([-limit, limit])
    axs.set_ylabel(r'$X$', fontsize=30)
    axs.set_xlabel(r'$Y$', fontsize=30)
    axs.tick_params(labelsize=20, length=0)

def plot_syrinx(obj, syll, axs, levels_, cmap, if_contour, contour_alpha=1, heatmap=False, colorbar=True):
    if obj.N_SYLL > 4:
        raise ValueError('Only 4 syllables are available in the syrinx landscape')
    if not heatmap and colorbar:
        print("Warning: Colorbar is only plotted when heatmap is True. Setting colorbar to False.")
        colorbar = False
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
        axs.contour(Z.T, levels=levels_, extent=[-1, 1, -1, 1], colors='k', linewidths=1, alpha=contour_alpha)
    if heatmap:
        cs = axs.contourf(Z.T, cmap=cmap, extent=[-1, 1, -1, 1], vmin=0, vmax=1, levels=levels_, alpha=contour_alpha)
        if colorbar:
            # Create colorbar on the inset axis (cax)
            cax = axs.inset_axes((1.05, 0, 0.08, 1.0))
            cbar = axs.figure.colorbar(cs, cax=cax)
            cbar.set_label('Performance Metric (R)', fontsize=30, rotation=270, labelpad=20)
            cbar.ax.tick_params(labelsize=20)
            cbar.ax.set_yticks([0, 1])
    axs.set_aspect('equal', adjustable='box')
    axs.set_xticks([-1, 1], [0, 1])
    axs.set_yticks([1], [0.2])
    axs.set_xlim([-limit, limit])
    axs.set_ylim([-limit, limit])
    axs.set_ylabel(r'$Pressure (P)$', fontsize=30)
    axs.set_xlabel(r'$Tension (T)$', fontsize=30)
    axs.tick_params(labelsize=20, length=0)
    # axs.scatter(target_pos[1], target_pos[0], s=100, c='green', marker='x', label='Target')


def plot_scatter_traj_helper(axs, x_traj, y_traj,
                             color_traj, alt_scatter_color, scatter_alpha=0.5,
                             plot_smooth_traj=False, running_smooth=20, traj_alpha=.8,
                             day_i=0, day_f=60, every_nth_point=2, TRIALS=1000,
                             label='motor output', daycolor=False, daycolorbar=False):


    if scatter_alpha > 0: 
        if daycolor:
            c_day = np.repeat(np.arange(day_i, day_f), TRIALS)

            cs = axs.scatter(
                x_traj[day_i * TRIALS: day_f * TRIALS][::every_nth_point],
                y_traj[day_i * TRIALS: day_f * TRIALS][::every_nth_point], 25, c = c_day[::every_nth_point], cmap='plasma', label=label, edgecolors='none', alpha=scatter_alpha, marker='.', zorder=5
            )
            
            if daycolorbar:
                cax = axs.inset_axes((1.05, 0, 0.08, 1.0))
                cbar = axs.figure.colorbar(cs, cax=cax)
                cbar.set_label('DPH', fontsize=30, rotation=270)#, labelpad=10)
                cbar.ax.tick_params(labelsize=20)
                cbar.ax.set_yticks([day_i, day_f])
                cbar.ax.set_yticklabels([day_i+40, day_f+40])

        else:   
            scatter_color = alt_scatter_color if plot_smooth_traj else color_traj
            axs.scatter(
                x_traj[day_i * TRIALS: day_f * TRIALS][::every_nth_point],
                y_traj[day_i * TRIALS: day_f * TRIALS][::every_nth_point], 25, color = scatter_color, label=label, edgecolors='none', alpha=scatter_alpha, marker='.', zorder=5
            )


    if not daycolor:
        axs.scatter(x_traj[0], y_traj[0],
                    s=150, c=color_traj,
                    marker='s', zorder=500)#, label='Starting Point')
        axs.scatter(x_traj[-1], y_traj[-1],
                    s=150, c='white',
                    marker='X', zorder=500)#, label='Ending Point')
        axs.scatter(x_traj[0], y_traj[0],
                    s=50, c='white',
                    marker='s', zorder=600, label=f'Initial {label}')
        axs.scatter(x_traj[-1], y_traj[-1],
                    s=50, c=color_traj,
                    marker='x', zorder=600, label=f'Final {label}')

    
    if plot_smooth_traj:
        xtraj_smooth = running_mean(np.array(x_traj), N = running_smooth)
        ytraj_smooth = running_mean(np.array(y_traj), N = running_smooth)
        
        X = xtraj_smooth[day_i * TRIALS: day_f * TRIALS]
        Y = ytraj_smooth[day_i * TRIALS: day_f * TRIALS]

        V = [np.stack([x, y]) for x, y in zip(X, Y)]
        V = np.array(V).reshape((1, len(X), 2))
        lines = LineCollection(V, color=color_traj, alpha=traj_alpha, linewidth=1, zorder=100)
        axs.add_collection(lines)

    

def plot_scatter_traj(obj, syll, day_i, day_f, every_nth_point,
                      plot_smooth_traj = False, smooth_window=20, traj_alpha=.8, figsize=(10,10), scatter_alpha = 0.5, plot_daily_start_points = False, if_contour=False, contour_alpha=1, heatmap=False, colorbar=False, legend=False, plot_motor=True, plot_cortex=False, plot_BG=False, daycolor=False, daycolorbar=False):
    fig, axs = plt.subplots(figsize=figsize)
    cmap = 'Greys'# color_contour_bckg # cmap param doesn't work # Match the colormap style from plot_landscape
    levels_ = 12 # 12 fix!
    # TRIALS = obj.TRIALS
    

    # Plot background landscape
    if obj.LANDSCAPE == 0:
        print("Plotting artificial landscape")
        plot_artificial(obj, syll, axs, levels_, cmap, if_contour=if_contour, contour_alpha=contour_alpha, heatmap=heatmap, colorbar=colorbar)
    else:
        print("Plotting syrinx landscape")
        plot_syrinx(obj, syll, axs, levels_, cmap, if_contour=if_contour, contour_alpha=contour_alpha, heatmap=heatmap, colorbar=colorbar)
    
    # Plot agent trajectory
    # axs.plot(
    #     x_traj[day_i * TRIALS: day_f * TRIALS][::every_nth_point],
    #     y_traj[day_i * TRIALS: day_f * TRIALS][::every_nth_point], color = color_motor, label='Agent Trajectory', alpha=.5, linewidth=0.0, marker='.', markersize=1
    # )
    # if plot_smooth_traj:
    #     xtraj_smooth = running_mean_dynamic(np.array(x_traj), N_i=N_i, N_f=N_f, steepness=steepness)
    #     ytraj_smooth = running_mean_dynamic(np.array(y_traj), N_i=N_i, N_f=N_f, steepness=steepness)
    #     axs.plot(
    #         xtraj_smooth[day_i * TRIALS: day_f * TRIALS], #[::every_nth_point],
    #         ytraj_smooth[day_i * TRIALS: day_f * TRIALS],#[::every_nth_point],
    #         color = color_motor, label='Motor Trajectory', alpha=.5, linewidth=0.5, marker='.', markersize=1, ls='-'
    #     )
    
    # axs.plot(
    #     x_traj[day_i * TRIALS: day_f * TRIALS][::every_nth_point],
    #     y_traj[day_i * TRIALS: day_f * TRIALS][::every_nth_point], color = 'white', label='Agent Trajectory', alpha=1, linewidth=0.0, marker='.', markersize=3, zorder=3
    # )
    # axs.plot(
    #     x_traj[day_i * TRIALS: day_f * TRIALS][::every_nth_point],
    #     y_traj[day_i * TRIALS: day_f * TRIALS][::every_nth_point], color = color_motor, label='Agent Trajectory', alpha=1, linewidth=0.0, marker='.', markersize=2, zorder=3
    # )

    if plot_motor:
        x_traj, y_traj = zip(*obj.actions[:, :, syll, :].reshape(-1, 2))

        plot_scatter_traj_helper(axs,
                                 x_traj, y_traj,
                                 color_motor,
                                 alt_scatter_color = 'grey',
                                 scatter_alpha=scatter_alpha,
                                 plot_smooth_traj=plot_smooth_traj,
                                 running_smooth=smooth_window,
                                 traj_alpha=traj_alpha,
                                 day_i=day_i, day_f=day_f, every_nth_point=every_nth_point,
                                 TRIALS=obj.TRIALS,
                                 label='motor output',
                                 daycolor=daycolor,
                                 daycolorbar=daycolorbar)



    if plot_cortex:
        ra_actions = obj.actions - obj.actions_bg  
        x_ra, y_ra = zip(*ra_actions[:, :, syll, :].reshape(-1, 2))

        plot_scatter_traj_helper(axs,
                                 x_ra, y_ra,
                                 color_cortical,
                                 alt_scatter_color=sns_cmap[-1],
                                 scatter_alpha=scatter_alpha,
                                 plot_smooth_traj=plot_smooth_traj,
                                 running_smooth=smooth_window,
                                 traj_alpha=traj_alpha,
                                 day_i=day_i, day_f=day_f, every_nth_point=every_nth_point,
                                 TRIALS=obj.TRIALS,
                                 label='cortical contribution',
                                 daycolor=daycolor,
                                 daycolorbar=daycolorbar)
    

    
    if plot_BG:
        bg_actions = obj.actions_bg   
        x_bg, y_bg = zip(*bg_actions[:, :, syll, :].reshape(-1, 2))
        
        plot_scatter_traj_helper(axs,
                                 x_bg, y_bg,
                                 color_bg,
                                 alt_scatter_color='gold',
                                 scatter_alpha=scatter_alpha,
                                 plot_smooth_traj=plot_smooth_traj,
                                 running_smooth=smooth_window,
                                 traj_alpha=traj_alpha,
                                 day_i=day_i, day_f=day_f, every_nth_point=every_nth_point,
                                 TRIALS=obj.TRIALS,
                                 label='BG contribution',
                                 daycolor=daycolor,
                                 daycolorbar=daycolorbar)

        if daycolor:
            axs.scatter(0, 0,
                        s=150, c='white',
                        marker='o', zorder=500)
            axs.scatter(0, 0,
                        s=50, c=color_cortical,
                        marker='o', zorder=600, label='Cortical contribution')
            


    if plot_daily_start_points:
        x = x_traj[day_i * TRIALS: day_f * TRIALS][0::1000]
        y = y_traj[day_i * TRIALS: day_f * TRIALS][0::1000]
        c_val = np.arange(len(x))
        axs.scatter(
            x, y, 200, c = 'white', label='Agent Trajectory', edgecolors='none', alpha=1, marker='.', zorder=900
        )
        axs.scatter(
            x, y, 150, c = c_val, label='Agent Trajectory', edgecolors='none', alpha=1, marker='.', zorder=1000, cmap='magma'
        )

    
    
    # Labels
    # axs.set_ylabel(r'$P$', fontsize=30)
    # axs.set_xlabel(r'$T$', fontsize=30)
    # axs.set_ylabel(r'$P_{\alpha}$ (Pressure)', fontsize=22)
    # axs.set_xlabel(r'$P_{\beta}$ (Tension)', fontsize=22)
    # axs.tick_params(labelsize=20)

    if legend:
        axs.legend(facecolor='lightgrey', bbox_to_anchor=(.8, 1.15), loc='upper left')#, edgecolor='black', framealpha=0.8)



    plt.tight_layout()

    # plt.savefig(figures_path+'contour_syll'+str(syll+1)+'.png')
    # plt.show()



def plot_landscape_only(obj, syll, contour_levels=12, contour_alpha=1, plot_colors = plot_colors, if_contour=False, heatmap=False, colorbar=False):
    fig, axs = plt.subplots(figsize=(12, 10))
    cmap = 'Greys' # LinearSegmentedColormap.from_list('change_this', ['white', 'white']) # 'Greys' color_contour_bckg #'Purples' #LinearSegmentedColormap.from_list('white_to_black', ['white', 'rebeccapurple'])
    levels_ = contour_levels
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
        plot_artificial(obj, syll, axs, levels_, cmap, contour_alpha=contour_alpha, if_contour=if_contour, heatmap=heatmap, colorbar=colorbar)
    else:
        print("Plotting syrinx landscape")
        plot_syrinx(obj, syll, axs, levels_, cmap, contour_alpha=contour_alpha, if_contour=if_contour, heatmap=heatmap, colorbar=colorbar)
    
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
    # axs.tick_params(labelsize=20)
    # axs.legend()
    plt.tight_layout()
    # plt.show()


###### PLOT MOTOR OUTPUTS ######
def plot_output(obj, syll, skip_size=1, window_size=10, plot_raw=True, plot_cortex=False, plot_BG=False, plot_alpha=1, figsize=None):
    figure, (ax1, ax2) = plt.subplots(2,1, figsize=figsize)

    N_SYLL = obj.N_SYLL # To plot only one syllable at a time, set N_SYLL to 1 and plot the first syllable (syll=0)
    N_DAILY_MOTIFS = obj.TRIALS
    DAYS = obj.DAYS
    sk = N_SYLL
    LIMIT = 1.5

    # Display x axis in days
    x = np.arange(DAYS*N_DAILY_MOTIFS)[::skip_size]
    # x = x/(N_DAILY_MOTIFS * N_SYLL)

    x_bg_traj, y_bg_traj = zip(*obj.actions_bg[:, :, syll, :].reshape(-1, 2))
    ra_actions = obj.actions - obj.actions_bg
    x_ra_traj, y_ra_traj = zip(*ra_actions[:, :, syll, :].reshape(-1, 2))
    x_bg_traj = np.array(x_bg_traj)
    y_bg_traj = np.array(y_bg_traj)
    x_ra_traj = np.array(x_ra_traj)
    y_ra_traj = np.array(y_ra_traj)
    

    # ax1.plot(obj.centers[syll, 1]*np.ones(N_DAILY_MOTIFS*DAYS),  color='black', linestyle='-', linewidth=1)
    # ax2.plot(obj.centers[syll, 0]*np.ones(N_DAILY_MOTIFS*DAYS),  color='black', linestyle='-', linewidth=1)

    if plot_raw:
        ax1.scatter(x, obj.actions[:,:,syll,0].reshape(DAYS*N_DAILY_MOTIFS)[::skip_size], s=1, color='grey', alpha=.2, marker='.', zorder=100)
        ax2.scatter(x, obj.actions[:,:,syll,1].reshape(DAYS*N_DAILY_MOTIFS)[::skip_size], s=1, color='grey', alpha=.2, marker='.', zorder=100)

    ax1.plot(running_mean(obj.actions[:,:,syll,0].reshape(DAYS*N_DAILY_MOTIFS), window_size), color=color_motor, lw=1, alpha=plot_alpha, zorder=101)
    ax2.plot(running_mean(obj.actions[:,:,syll,1].reshape(DAYS*N_DAILY_MOTIFS), window_size), color=color_motor, lw=1, alpha=plot_alpha, zorder=101)



    if plot_cortex:
        if plot_raw:
            ax1.scatter(x, x_ra_traj[::skip_size], s=1, color=color_cortical, alpha=.2, marker='.', zorder=50)
            ax2.scatter(x, y_ra_traj[::skip_size], s=1, color=color_cortical, alpha=.2, marker='.', zorder=50)

        ax1.plot(running_mean(x_ra_traj, window_size), color=color_cortical, lw=1, alpha=plot_alpha, zorder=50)
        ax2.plot(running_mean(y_ra_traj, window_size), color=color_cortical, lw=1, alpha=plot_alpha, zorder=50)

    if plot_BG:
        if plot_raw:
            ax1.scatter(x, x_bg_traj[::skip_size], s=1, color=color_bg, alpha=.2, marker='.', zorder=50)
            ax2.scatter(x, y_bg_traj[::skip_size], s=1, color=color_bg, alpha=.2, marker='.', zorder=50)

        ax1.plot(running_mean(x_bg_traj, window_size), color=color_bg, lw=1, alpha=plot_alpha, zorder=50)
        ax2.plot(running_mean(y_bg_traj, window_size), color=color_bg, lw=1, alpha=plot_alpha, zorder=50)

    # Plot running average of cortical output (brown), BG output (grey) and total output (black)
    # Data
    # for syll in range(N_SYLL):
        # motor outputs
    # ax1.plot(running_mean(obj.actions[:,:,syll,0].reshape(DAYS*N_DAILY_MOTIFS), window_size), color=color_motor, lw=0, alpha=1, marker=',')
    # ax2.plot(running_mean(obj.actions[:,:,syll,1].reshape(DAYS*N_DAILY_MOTIFS), window_size), color=color_motor, lw=0, alpha=1, marker=',')

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
    
    ax1.plot(obj.centers[syll, obj.LANDSCAPE*1]*np.ones(N_DAILY_MOTIFS*DAYS),  color='grey', linestyle='--', linewidth=2, label = 'Global optimum', zorder=500)
    ax2.plot(obj.centers[syll, 1-obj.LANDSCAPE]*np.ones(N_DAILY_MOTIFS*DAYS),  color='grey', linestyle='--', linewidth=2, zorder=500)


    # Axis beauty
    # ax1.axvline(x=N_DAYS_INTACT * N_DAILY_MOTIFS, linestyle='--', color='grey', lw=1)
    # ax2.axvline(x=N_DAYS_INTACT * N_DAILY_MOTIFS, linestyle='--', color='grey', lw=1)
    # ax1.axhline(y=0, linestyle='--', color='black', alpha=0.1)
#         ax1.axhline(y=obj.targetpos[0], linestyle='--', color='black', label='Global optimum')
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.get_xaxis().set_ticks([])
    ax1.set_ylim(-LIMIT, LIMIT)
    ax1.tick_params(labelsize=15)
    ax1.set_yticks([-LIMIT, LIMIT], [0, 1] )
    ax1.set_xlim(-N_DAILY_MOTIFS, N_DAILY_MOTIFS*(DAYS))
    
    # Move y-axes to the right side
    for ax in (ax1, ax2):
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

    ax2.set_xlabel('DPH', fontsize=20)
    ax2.set_yticks([-LIMIT, LIMIT], [0, 0.2] )
    ax2.set_ylim(-LIMIT, LIMIT)
    ax2.tick_params(labelsize=15)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.set_xlim(-N_DAILY_MOTIFS, N_DAILY_MOTIFS*DAYS)
    # ax2.spines['bottom'].set_bounds(0, obj.n_days+obj.n_lesioned_days)
    
    if obj.LANDSCAPE == 1:
        ax2.set_ylabel(r'$P$', fontsize=20, labelpad=-10, rotation=270)
        ax1.set_ylabel(r'$T$', fontsize=20, rotation=270)
        ax1.set_yticks([-LIMIT, LIMIT], [0, 1] )
        ax2.set_yticks([-LIMIT, LIMIT], [0, 0.2] )
    else:
        ax2.set_ylabel(r'$Y$', fontsize=20, rotation=270, labelpad=15)
        ax1.set_ylabel(r'$X$', fontsize=20, rotation=270, labelpad=15)
        ax1.set_yticks([-LIMIT, 0, LIMIT], [-1, 0, 1])
        ax2.set_yticks([-LIMIT, 0, LIMIT], [-1, 0, 1])
    plt.legend(frameon=False, loc='center right', fontsize=12, bbox_to_anchor=(1.03,1))
    plt.xticks(range(0, N_DAILY_MOTIFS*(DAYS+1), 20*N_DAILY_MOTIFS), np.arange(40, DAYS+1+40, 20))
    ax1.legend()
    ax1.legend().get_frame().set_facecolor('lightgray')
    plt.tight_layout()
    # plt.savefig(figures_path+'motor_output'+'.png')
    # plt.show()


def plot_position_change_helper(x1, y1,
                                x2, y2,
                                axs,
                                color1, color2,
                                alpha=0.5, ls='-',
                                label1=None, label2=None,
                                legend=False):
    
    x = x1
    y = y1
    c_val = np.arange(len(x))
    axs.scatter(
        x, y, 150, c = c_val, label=label1, edgecolors='none', alpha=alpha, marker='.', zorder=200, cmap='plasma'
    )

    x = x2
    y = y2
    c_val = np.arange(len(x))
    axs.scatter(
        x, y, 150, c = color2, label=label2, edgecolors='none', alpha=alpha, marker='.', zorder=200, cmap='plasma'
    )


    # Connect start and end points for each day
    V = np.array([[[x1[i], y1[i]], [x2[i], y2[i]]] 
                for i in range(len(x1))])
    lines = LineCollection(V, array=c_val, cmap='plasma', alpha=alpha, linewidth=2, ls=ls)
    axs.add_collection(lines)

    if legend:
        axs.legend(facecolor='lightgrey')#, edgecolor='black', framealpha=0.8)


    plt.tight_layout()




def plot_position_change(obj, syll, day_i, day_f,
                      figsize=(10,10), alpha = 0.5,
                      if_contour=False, contour_alpha=1,
                      heatmap=False, colorbar=False, legend=False,
                      plot_motor=True, plot_cortex=False, plot_BG=False,
                      day_change=True, night_change=True):
                    # , plot_cortex=False, plot_BG=False,
                    #   daycolor=False, daycolorbar=False):
    fig, axs = plt.subplots(figsize=figsize)
    cmap = 'Greys'# color_contour_bckg # cmap param doesn't work # Match the colormap style from plot_landscape
    levels_ = 12 # 12 fix!
    TRIALS = obj.TRIALS
    

    # Plot background landscape
    if obj.LANDSCAPE == 0:
        print("Plotting artificial landscape")
        plot_artificial(obj, syll, axs, levels_, cmap, if_contour=if_contour, contour_alpha=contour_alpha, heatmap=heatmap, colorbar=colorbar)
    else:
        print("Plotting syrinx landscape")
        plot_syrinx(obj, syll, axs, levels_, cmap, if_contour=if_contour, contour_alpha=contour_alpha, heatmap=heatmap, colorbar=colorbar)
    


    color_day = 'orange'
    color_night = 'black'


    if plot_motor:
        x_traj, y_traj = zip(*obj.actions[:, :, syll, :].reshape(-1, 2))

    
    if plot_cortex:
        ra_actions = obj.actions - obj.actions_bg  
        x_traj, y_traj = zip(*ra_actions[:, :, syll, :].reshape(-1, 2))
    

    if plot_BG:
        bg_actions = obj.actions_bg  
        x_traj, y_traj = zip(*bg_actions[:, :, syll, :].reshape(-1, 2))
    


    x_day_start = x_traj[day_i * TRIALS: day_f * TRIALS][0::TRIALS]
    y_day_start = y_traj[day_i * TRIALS: day_f * TRIALS][0::TRIALS]

    x_day_end = x_traj[day_i * TRIALS: day_f * TRIALS][TRIALS-1::TRIALS]
    y_day_end = y_traj[day_i * TRIALS: day_f * TRIALS][TRIALS-1::TRIALS]

    # Plot position change from start to end of day
    if day_change:    
        plot_position_change_helper(x_day_start, y_day_start,
                                x_day_end, y_day_end,
                                axs,
                                color_day, color_night,
                                alpha=alpha,
                                ls='-',
                                label1='Start of day', label2='End of day',
                                legend=legend)
    
    # and from end of preceding night to start of day
    if night_change:
        plot_position_change_helper(
                                x_day_start[1:], y_day_start[1:],
                                x_day_end[:-1], y_day_end[:-1],
                                axs,
                                color_day, color_night,
                                alpha=alpha,
                                ls='--',
                                label1='Start of day'*(day_change==0), label2='End of preceding night'*(day_change==0),
                                legend=legend)


    
def plot_position_change_helper(x1, y1,
                                x2, y2,
                                axs,
                                color1, color2,
                                alpha=0.5, ls='-',
                                label1=None, label2=None,
                                legend=False):
    
    x = x1
    y = y1
    c_val = np.arange(len(x))
    axs.scatter(
        x, y, 150, c = c_val, label=label1, edgecolors='none', alpha=alpha, marker='.', zorder=200, cmap='plasma'
    )

    x = x2
    y = y2
    c_val = np.arange(len(x))
    axs.scatter(
        x, y, 150, c = color2, label=label2, edgecolors='none', alpha=alpha, marker='.', zorder=200, cmap='plasma'
    )


    # Connect start and end points for each day
    V = np.array([[[x1[i], y1[i]], [x2[i], y2[i]]] 
                for i in range(len(x1))])
    lines = LineCollection(V, array=c_val, cmap='plasma', alpha=alpha, linewidth=2, ls=ls)
    axs.add_collection(lines)

    if legend:
        axs.legend(facecolor='lightgrey')#, edgecolor='black', framealpha=0.8)


    plt.tight_layout()


    
def plot_jump_size_over_time_helper(x1,
                                x2,
                                axs,
                                color,
                                alpha=0.9, ls='-',
                                euclidean=False,
                                label_legend=None,
                                label_y=None,
                                legend=False,
                                print_initial_jump_size=False,
                                print_final_jump_size=False):

    if euclidean:
        y1 = np.array(x1[1])   
        x1 = np.array(x1[0])
        y2 = np.array(x2[1])
        x2 = np.array(x2[0])
        jump = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    else:
        x1 = np.array(x1[0])
        x2 = np.array(x2[0])
        jump = np.abs(x2 - x1)


    x = np.arange(jump.shape[0])
    if label_legend is 'Night change':
        x = x + 0.5 # Shift night change points to the right for better visibility
    axs.scatter(
        x, jump, 150, c = color, label=label_legend, edgecolors='none', alpha=alpha, marker='.'
    )
    if print_initial_jump_size:
        print(f"Initial jump size: {jump[0:5].mean():.3f}")
    if print_final_jump_size:
        print(f"Final jump size: {jump[-5:].mean():.3f}")

    if legend:
        axs.legend(facecolor='lightgrey')#, edgecolor='black', framealpha=0.8)

    axs.set_xlabel('DPH', fontsize=20)
    axs.set_ylabel(label_y+' jump size', fontsize=20)
    axs.set_xticks(range(0, int(jump.shape[0]+2), 10))
    axs.set_xticklabels(range(40, 40+int(jump.shape[0]+2), 10))
    axs.set_yticks(range(0, int(np.ceil(jump.max()+1))))
    axs.set_xlim(-1, jump.shape[0]+1)
    axs.spines['left'].set_bounds(0, int(np.ceil(jump.max())))
    axs.spines['bottom'].set_bounds(0, int(jump.shape[0]+1))
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    plt.tight_layout()




def plot_jump_size_over_time(obj, syll, day_i, day_f,
                      figsize=(10,4), alpha = 0.5,
                      legend=False,
                      plot_motor=True, plot_cortex=False, plot_BG=False, plot_reward=False,
                      day_change=True, night_change=False, start_rendition_change=False, end_rendition_change=False):
                    # , plot_cortex=False, plot_BG=False,
                    #   daycolor=False, daycolorbar=False):
    figure_size = (figsize[0], figsize[1]*(plot_motor + plot_cortex + plot_BG + plot_reward)) # Adjust height of figure based on number of subplots
    fig, axs = plt.subplots(plot_motor+plot_cortex+plot_BG+plot_reward, 1, figsize=figure_size)
    
    if plot_motor + plot_cortex + plot_BG + plot_reward == 1:
        axs = [axs]

    cmap = 'Greys'# color_contour_bckg # cmap param doesn't work # Match the colormap style from plot_landscape
    levels_ = 12 # 12 fix!
    TRIALS = obj.TRIALS
    
    color_day = 'orange'
    color_night = 'black'
    color_rendition_start = 'purple'
    color_rendition_end = 'red'
    
    types = ['Motor', 'Cortex', 'BG', 'Reward']
    plot_flags = [plot_motor, plot_cortex, plot_BG, plot_reward]

    for i, type in enumerate(types):
        print(type)
        if type=='Motor' and plot_motor:
            x_traj, y_traj = zip(*obj.actions[:, :, syll, :].reshape(-1, 2))
            euclidean = True

        if type=='Cortex' and plot_cortex:
            ra_actions = obj.actions - obj.actions_bg  
            x_traj, y_traj = zip(*ra_actions[:, :, syll, :].reshape(-1, 2))
            euclidean = True
        
        if type=='BG' and plot_BG:
            bg_actions = obj.actions_bg
            x_traj, y_traj = zip(*bg_actions[:, :, syll, :].reshape(-1, 2))
            euclidean = True
            
        if type=='Reward' and plot_reward:
            rewards = obj.rewards  
            x_traj = rewards[:, :, syll].reshape(-1, 1)
            # Just a duplicate of x_traj to use the same plotting function, since reward is 1D
            y_traj = rewards[:, :, syll].reshape(-1, 1)
            euclidean = False
            


        x_day_start = x_traj[day_i * TRIALS: day_f * TRIALS][0::TRIALS]
        y_day_start = y_traj[day_i * TRIALS: day_f * TRIALS][0::TRIALS]

        x_day_end = x_traj[day_i * TRIALS: day_f * TRIALS][TRIALS-1::TRIALS]
        y_day_end = y_traj[day_i * TRIALS: day_f * TRIALS][TRIALS-1::TRIALS]


        x_day_start_1 = x_traj[day_i * TRIALS: day_f * TRIALS][1::TRIALS]
        y_day_start_1 = y_traj[day_i * TRIALS: day_f * TRIALS][1::TRIALS]



        x_day_end_1 = x_traj[day_i * TRIALS: day_f * TRIALS][TRIALS-2::TRIALS]
        y_day_end_1 = y_traj[day_i * TRIALS: day_f * TRIALS][TRIALS-2::TRIALS]


        # Plot position change from start to end of day
        if day_change:   
 
            plot_jump_size_over_time_helper([x_day_start, y_day_start],
                                    [x_day_end, y_day_end],
                                    axs[i],
                                    color_day,
                                    alpha=alpha,
                                    ls='-',
                                    euclidean=euclidean,
                                    label_legend='Day change',
                                    label_y=type,
                                    legend=legend*(i==0))
        
        # and from end of preceding night to start of day
        if night_change:
            plot_jump_size_over_time_helper([x_day_end, y_day_end],
                                    [x_day_end_1, y_day_end_1],
                                    axs[i],
                                    color_night,
                                    alpha=alpha,
                                    ls='--',
                                    euclidean=True,
                                    label_legend='Night change',
                                    label_y=type,
                                    legend=legend*(i==0), 
                                    print_initial_jump_size=True,
                                    print_final_jump_size=True)



        # and from end of preceding night to start of day
        if start_rendition_change:
            plot_jump_size_over_time_helper([x_day_start, y_day_start],
                                    [x_day_start_1, y_day_start_1],
                                    axs[i],
                                    color_rendition_start,
                                    alpha=alpha,
                                    ls='-',
                                    euclidean=True,
                                    label_legend='Single rendition change day start',
                                    label_y=type,
                                    legend=legend*(i==0))

        # and from end of preceding night to start of day
        if end_rendition_change:
            plot_jump_size_over_time_helper([x_day_end, y_day_end],
                                    [x_day_end_1, y_day_end_1],
                                    axs[i],
                                    color_rendition_end,
                                    alpha=alpha,
                                    ls='-',
                                    euclidean=True,
                                    label_legend='Single rendition change day end',
                                    label_y=type,
                                    legend=legend*(i==0))

def plot_weights_unsorted(obj, type, chosen_syll=0):
    figure, (ax2, ax1) = plt.subplots(2, 1, figsize=(20, 6))

    if type=='HVC-RA':
        weights = obj.hvc_ra_array_all[:,:,chosen_syll,chosen_syll,:]
    if type=='HVC-BG':
        weights = obj.hvc_bg_array_all[:,:,chosen_syll,chosen_syll,:]

    post_synaptic_layer_size = weights.shape[-1]
    DAYS = obj.DAYS
    N_DAILY_MOTIFS = obj.TRIALS


    # ax1.axvline(x=N_DAYS_INTACT * N_DAILY_MOTIFS, linestyle='--', color='grey', lw=1)
    # ax2.axvline(x=N_DAYS_INTACT * N_DAILY_MOTIFS, linestyle='--', color='grey', lw=1)
    # cm = plt.cm.get_cmap('RdGy_r')
    if type=='HVC-RA':
        color = color_cortical
    if type=='HVC-BG':
        color = color_bg

    cm = LinearSegmentedColormap.from_list('grad', ['white', color])

    plot_array1 = weights.reshape(DAYS*N_DAILY_MOTIFS, post_synaptic_layer_size)
    im1 = ax1.imshow((plot_array1[:,:].T), cmap=cm, aspect='auto', interpolation='none', vmin = -1, vmax = 1)

    plot_array2 = weights.reshape(DAYS*N_DAILY_MOTIFS, post_synaptic_layer_size)
    ax2.plot(np.abs(plot_array2[::1, :]), color=color, alpha=.5, linewidth=1)

    figure.subplots_adjust(right=1.3)
    ax1_pos = ax1.get_position()
    cbar_ax = figure.add_axes([1, ax1_pos.y0+.02, .01, ax1_pos.height-.01])
    cbar = figure.colorbar(im1, cax=cbar_ax)
    cbar.set_label('Synaptic strength', fontsize=15, rotation=270, labelpad=15)
    cbar.set_ticks([-1, 1])

    
    # figure.subplots_adjust(right=1.2)
    # cbar_ax = figure.add_axes([1.05, 0.16, 0.03, 0.35]) # type:ignore
    # cbar = figure.colorbar(im1, cax=cbar_ax)
    # cbar.set_label('Activity level', fontsize=15)
    # cbar.set_ticks([0, 1])

    # ax1.vlines(BG_INTACT_DAYS * N_DAILY_MOTIFS, 0, 8, color='grey', linestyle='--', lw=1)   
    # ax2.vlines(BG_INTACT_DAYS * N_DAILY_MOTIFS, 0, 8, color='grey', linestyle='--', lw=1)
    # # ax1.axhline(y=0, linestyle='--', color='black', alpha=0.1)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_xticks([0, 20*N_DAILY_MOTIFS, 40*N_DAILY_MOTIFS, 60*N_DAILY_MOTIFS], np.arange(40, 40 + DAYS+1, 20))
    ax1.tick_params(labelsize=15)
    ax1.set_yticks([0, post_synaptic_layer_size])
    ax1.set_xlim(-N_DAILY_MOTIFS, DAYS*N_DAILY_MOTIFS)
    # ax1.set_ylim(-0.5, 7.5)
    ax1.set_xlabel('DPH', fontsize=20)
    ax1.spines['bottom'].set_bounds(0, DAYS*N_DAILY_MOTIFS)
    ax1.spines['left'].set_bounds(0, post_synaptic_layer_size)

    # ax2.set_ylim(-0.5, 7.5)
    ax2.set_xlabel('DPH', fontsize=20)
    ax2.set_yticks([0, 1])
    ax2.spines['left'].set_bounds(0, 1)
    ax2.set_xticks([0, 20*N_DAILY_MOTIFS, 40*N_DAILY_MOTIFS, 60*N_DAILY_MOTIFS], np.arange(40, 40 + DAYS+1, 20))
    ax2.set_xlim(-N_DAILY_MOTIFS, DAYS * N_DAILY_MOTIFS)
    ax2.tick_params(labelsize=15)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_bounds(0, DAYS*N_DAILY_MOTIFS) 

    if type=='HVC-RA':
        ax1.set_ylabel('HVC-RA\nsynapse', fontsize=20, labelpad=-20)
        ax2.set_ylabel('HVC-RA\nsynaptic\nstrength', fontsize=20)
    
    if type=='HVC-BG':
        ax1.set_ylabel('HVC-BG\nsynapse', fontsize=20, labelpad=-20)
        ax2.set_ylabel('HVC-BG\nsynaptic\nstrength', fontsize=20)

    plt.tight_layout()



def save_figure(filename, format="pdf", save=False, dpi=150, rasterized=True, metadata=None):
    if save:
        os.makedirs("Plots", exist_ok=True)
        if format=='.pdf' or format=='.svg': rasterized = True
        if format=='.png': rasterized = False
        if rasterized:
            fig = plt.gcf()
            for ax in fig.get_axes():
                ax.set_rasterized(True)
        plt.savefig(os.path.join("Plots", f"{filename}.{format}"), dpi=dpi, bbox_inches="tight", metadata=metadata)
    

    