#!/bin/bash
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=6G        # memory per node
#SBATCH --time=2-00:00      # time (DD-HH:MM)
#SBATCH --output=job_logs/maze_sd_sweep/%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-lelis
#SBATCH --mail-user=arajabpo@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-9


source ~/Sparse-Policies/venv/bin/activate

module load flexiblas
export FLEXIBLAS=imkl

python  ~/Sparse-Policies/Code/scripts/train_ppo.py \
--env_id Karel_maze \
--seed $SLURM_ARRAY_TASK_ID \
--game_width 8 \
--game_height 8 \
--max_steps 100 \
--num_steps 300 \
--sparse_reward \
--hidden_size 32 \
--total_timesteps 2_000_000 \
--num_envs 1 \
--num_minibatches 1 \
--ppo_type original \
--multi_initial_confs \
--l1_lambda 0.00001 \
--learning_rate 0.001 \
--ent_coef 0.01 \
--clip_coef 0.2 \
--exp_name maze_F3_MS100_ent.01_sweep