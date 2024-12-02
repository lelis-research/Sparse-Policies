#!/bin/bash
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=1G        # memory per node
#SBATCH --time=2-00:00      # time (DD-HH:MM)
#SBATCH --output=job_logs/karel/%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-lelis


source ~/Sparse-Policies/venv/bin/activate

module load flexiblas
export FLEXIBLAS=imkl

python  ~/Sparse-Policies/Code/environment/karel_env/gym_envs/karel_gym.py