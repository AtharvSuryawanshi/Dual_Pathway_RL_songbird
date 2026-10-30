# testing between cluster of 10 and 2

import numpy as np
import os 
import matplotlib.pyplot as plt
save_dir = "plots"
TEST_NOS = 10
seeds = np.random.randint(0, 500, TEST_NOS)
seeds = np.sort(seeds)
print("Seeds: ", seeds)

from module import build_and_run
from module10 import build_and_run10
from tqdm import tqdm

cluster2_returns = []
cluster10_returns = []

for filename in os.listdir(save_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        os.remove(os.path.join(save_dir, filename))  
        
for i in tqdm(range(TEST_NOS)):
    seed = seeds[i]
    returns = build_and_run(seed, True, True)
    cluster2_returns.append(returns)
    returns = build_and_run10(seed, True, True)
    cluster10_returns.append(returns)
print(f"Cluster 2= {cluster2_returns} ")
print(f"Cluster 10 = {cluster10_returns}")
fig, ax = plt.subplots(1,1) 
ax.plot(cluster2_returns, label='Cluster 2', linewidth=0, marker='o')
ax.plot(cluster10_returns, label='Cluster 10', linewidth=0, marker='o')
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
