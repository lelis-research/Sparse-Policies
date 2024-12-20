#!/bin/bash
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=1G        # memory per node
#SBATCH --time=00-00:90      # time (DD-HH:MM)
#SBATCH --output=job_logs/sweep_xavier_noSparse/%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-lelis
#SBATCH --mail-user=arajabpo@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0


source ~/Sparse-Policies/venv/bin/activate

module load flexiblas
export FLEXIBLAS=imkl


python  ~/Sparse-Policies/Code/scripts/run_sweep.py \
--env_id Karel_stair_climber \
--seed $SLURM_ARRAY_TASK_ID \
--game_width 12 \
--game_height 12 \
--max_steps 50 \
--num_steps 300 \
--sparse_reward \
--hidden_size 32 \
--total_timesteps 200 \
--num_envs 1 \
--num_minibatches 1 \
--ppo_type gru \
--value_learning_rate 0.05 \
--exp_name stairClimber_PPO_GRU_Sparse_RPosEx_sweep_test