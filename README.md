# Dual Pathway Architecture for songlearning in zebra finches
## 🛠 Getting Started


## Installation
Set up your environment with these simple steps:

Clone the repository:

```bash
# Clone the repo 
git clone -b library https://github.com/AtharvSuryawanshi/Dual_Pathway_RL_songbird.git

# Create and activate environment
conda create --name dual_pathway_env python=3.12
conda activate dual_pathway_env

# Install dependencies
cd Dual_Pathway_RL_songbird
pip install -r requirements.txt

# Install dual_pathway_model package
pip install -e .
```

## Repository Structure

```text
Dual_Pathway_RL_songbird/
|-- src/dual_pathway_model/
|   |-- model.py                # Core model definition
|   |-- functions.py            # Helper functions
|   |-- directory_functions.py  # Directory helper functions
|   `-- params_base.py          # Default parameters
|-- Robustness/
|   |-- results/
            `-- PARAM_NAME 
                    |-- meta.yaml                       # Metadata from experiment
                    `-- terminal_performance.npy        # Terminal performance 
|   `-- robustness.py           # Main simulation script
|-- Figures/
|   `-- Figure_x.ipynb
`-- requirements.txt          # package requirements
```


