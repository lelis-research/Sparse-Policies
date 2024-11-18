#!/bin/bash
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=1G        # memory per node
#SBATCH --time=2-00:00      # time (DD-HH:MM)
#SBATCH --output=job_logs/%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-lelis

source ~/Sparse-Policies/venv/bin/activate
pip install seaborn plotly

python  ~/Sparse-Policies/Code/evaluation/plot_experiment_results.py