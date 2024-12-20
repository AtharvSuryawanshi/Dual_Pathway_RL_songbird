import numpy as np
import os 
import matplotlib.pyplot as plt
save_dir = "plots"
TEST_NOS = 5
# seeds = np.random.randint(0, 500, TEST_NOS)
# seeds = np.sort(seeds)
seeds = [233, 275, 290, 311, 402]
print("Seeds: ", seeds)

from module import build_and_run
from tqdm import tqdm

annealing_returns = []
non_annealing_returns = []

for filename in os.listdir(save_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        os.remove(os.path.join(save_dir, filename))  
        
for i in tqdm(range(TEST_NOS)):
    seed = seeds[i]
    returns = build_and_run(seed, True, True)
    annealing_returns.append(returns)
    returns = build_and_run(seed, False, True)
    non_annealing_returns.append(returns)
print(f"Annealing = {annealing_returns} ")
print(f"Non annealing = {non_annealing_returns}")  



fig, ax = plt.subplots(1,1) 
ax.plot(annealing_returns, label='Annealing', linewidth=0, marker='o')
ax.plot(non_annealing_returns, label='Non-Annealing', linewidth=0, marker='o')
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
# plt.show()
