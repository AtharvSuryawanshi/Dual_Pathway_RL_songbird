import numpy as np
import sys
import os 
import matplotlib.pyplot as plt
from module import *
from tqdm import tqdm

save_dir = "plots"
TEST_NOS = 10
seeds = np.random.randint(0, 500, TEST_NOS)
seeds = np.sort(seeds)
# seeds = [233, 275, 290, 311, 402]

# Redirect stdout to a file
original_stdout = sys.stdout  # Save a reference to the original standard output
output_file_path = os.path.join(save_dir, "output.txt")
with open(output_file_path, 'w') as f:
    sys.stdout = f  # Change the standard output to the file we created.
    print("Seeds: ", seeds)

    non_annealing_returns = []
    annealing_1 = []
    annealing_2 = []
    annealing_3 = []
    annealing_4 = []

    for filename in os.listdir(save_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            os.remove(os.path.join(save_dir, filename))

    for i in tqdm(range(TEST_NOS)):
        seed = seeds[i]
        returns = build_and_run(seed, False, True)
        non_annealing_returns.append(returns)
    print(f"Non annealing = {non_annealing_returns}")

    # annealing 1
    RA_sig_slope = 18
    LEARNING_RATE_HL = 1.6e-5
    for i in tqdm(range(TEST_NOS)):
        seed = seeds[i]
        returns = build_and_run(seed, True, True)
        annealing_1.append(returns)
    print(f"RA_sig_slope = {RA_sig_slope}, LEARNING_RATE_HL = {LEARNING_RATE_HL}")
    print(f"Annealing 1 = {annealing_1}")

    # annealing 2
    RA_sig_slope = 15
    LEARNING_RATE_HL = 1.6e-5
    for i in tqdm(range(TEST_NOS)):
        seed = seeds[i]
        returns = build_and_run(seed, True, True)
        annealing_2.append(returns)
    print(f"RA_sig_slope = {RA_sig_slope}, LEARNING_RATE_HL = {LEARNING_RATE_HL}")
    print(f"Annealing 2 = {annealing_2}")

    # annealing 3
    RA_sig_slope = 18
    LEARNING_RATE_HL = 2e-5
    for i in tqdm(range(TEST_NOS)):
        seed = seeds[i]
        returns = build_and_run(seed, True, True)
        annealing_3.append(returns)
    print(f"RA_sig_slope = {RA_sig_slope}, LEARNING_RATE_HL = {LEARNING_RATE_HL}")
    print(f"Annealing 3 = {annealing_3}")

    # annealing 4
    RA_sig_slope = 15
    LEARNING_RATE_HL = 1.6e-5
    for i in tqdm(range(TEST_NOS)):
        seed = seeds[i]
        returns = build_and_run(seed, True, True)
        annealing_4.append(returns)
    print(f"RA_sig_slope = {RA_sig_slope}, LEARNING_RATE_HL = {LEARNING_RATE_HL}")
    print(f"Annealing 4 = {annealing_4}")

    sys.stdout = original_stdout  # Reset the standard output to its original value

# Plotting is done outside the file writing block to not capture plot generation messages
fig, ax = plt.subplots(1,1)
ax.plot(non_annealing_returns, label='Non-Annealing', linewidth=0, marker='o')
ax.plot(annealing_1, label='Annealing 1', linewidth=0, marker='o')
ax.plot(annealing_2, label='Annealing 2', linewidth=0, marker='o')
ax.plot(annealing_3, label='Annealing 3', linewidth=0, marker='o')
ax.plot(annealing_4, label='Annealing 4', linewidth=0, marker='o')
ax.set_ylim(0, 1)
plt.xticks(np.arange(0, TEST_NOS, step=1), labels=seeds)
ax.legend()
ax.set_xlabel('Seeds')
ax.set_ylabel('Returns')
plt.hlines(0.7, 0, TEST_NOS, linestyles='dashed')
fig.suptitle('Results', fontsize=20)

# Save the plot
plt.savefig(os.path.join(save_dir, f"Overall_results.png"))
plt.close()  # Close the plot to avoid memory leaks