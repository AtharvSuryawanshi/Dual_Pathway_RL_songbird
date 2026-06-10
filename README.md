# Dual pathway architecture in songbirds enables robust sensorimotor learning

A reinforcement learning model implementing dual-pathway architecture for birdsong learning, inspired by the songbird brain. Codes for the research at https://doi.org/10.64898/2026.05.07.723469

## 🛠 Getting Started

### Installation

Set up your environment with these simple steps:

```bash
# Clone the repo 
git clone https://github.com/AtharvSuryawanshi/Dual_Pathway_RL_songbird.git

# Create and activate environment
conda create --name dual_pathway_env python=3.12
conda activate dual_pathway_env

# Install dependencies and package
cd Dual_Pathway_RL_songbird
pip install -r requirements.txt -e .
```

## Repository Structure

```text
Dual_Pathway_RL_songbird/
├── dual_pathway_model/                      # Core model implementation
│   ├── model.py                             # Main dual-pathway model
│   ├── model_SA.py                          # Simulated annealing variant
│   ├── env_lite.py                          # Lightweight environment
│   ├── functions.py                         # Utility functions
│   ├── directory_functions.py               # Directory management utilities
│   ├── plotting_functions.py                # Visualization functions
│   ├── __init__.py                          # Package initialization
│   ├── params.yaml                          # Default model parameters
│   ├── params_SA.yaml                       # Simulated annealing parameters
│   └── plotting_colors.yaml                 # Color scheme definitions
│
├── Robustness/                              # Robustness analysis experiments
│   ├── robustness.py                        # Robustness testing script
│   ├── robustness_params.yaml               # Experiment parameters
│   ├── analysis.ipynb                       # Analysis and results visualization
│   └── results/                             # Experiment results
│       └── {PARAM_NAME}/
│           ├── meta.yaml                    # Experiment metadata
│           └── terminal_performance.npy     # Terminal performance data
│
├── Benchmarks/                              # Model comparison experiments
│   ├── benchmarks.py                        # Benchmark comparison script
│   ├── benchmark_params.yaml                # Benchmark parameters
│   ├── analysis.ipynb                       # Results analysis notebook
│   ├── models/                              # Reference model implementations
│   ├── Plots/                               # Benchmark visualization outputs
│   └── results/                             # Benchmark experiment results
│
├── Lesion_exp/                              # Brain lesioning experiments
│   ├── lesion.py                            # Lesioning experiment script
│   ├── lesion_params.yaml                   # Lesion experiment parameters
│   ├── analysis.ipynb                       # Analysis and visualization
│   ├── models/                              # Model variants
│   └── results/                             # Lesion experiment results
│
├── Stat_test/                               # Statistical testing
│
├── Figures/                                 # Publication figures
│   └── Figure_*.ipynb                       # Figure generation notebooks
│
├── Archives/                                # Legacy and historical code
│
├── pyproject.toml                           # Project configuration
├── requirements.txt                         # Python package requirements
├── LICENSE                                  # License file
└── README.md                                # This file
```



## Directory Descriptions

| Directory | Purpose |
|-----------|---------|
| `dual_pathway_model/` | Core model implementation and utilities |
| `Robustness/` | Parameter sensitivity and robustness testing |
| `Benchmarks/` | Comparative analysis with other models |
| `Lesion_exp/` | Targeted ablation studies of brain areas during learning |
| `Stat_test/` | Statistical testing and validation |
| `Figures/` | Jupyter notebooks for generating publication figures |
| `Archives/` | Historical code and previous implementations |
