#!/bin/bash

cd DevRL
/home/asuryawanshi/miniforge3/bin/python run_bench.py
cd ..

cd Dual_pathway
/home/asuryawanshi/miniforge3/bin/python run_bench.py
cd ..

cd Sim_Annealing
/home/asuryawanshi/miniforge3/bin/python run_bench.py
cd ..

cd std_RL
/home/asuryawanshi/miniforge3/bin/python run_bench.py
cd ..