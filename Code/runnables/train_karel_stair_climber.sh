#!/bin/bash
#SBATCH --cpus-per-task=32   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16G        # memory per node
#SBATCH --time=2-00:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-lelis
#SBATCH --mail-user=arajabpo@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-5


source /home/iprnb/venvs/neural-policy-decomposition/bin/activate
source /home/amirrj/projects/def-lelis/amirrj/Sparse-Policies/Sparse-Policies/venv/bin/activate
module load python/3.11

python3.11  /home/iprnb/projects/def-lelis/iprnb/neural-policy-decomposition/agent_recurrent.py --rnn_type "gru" --hidden_size 64 --l1_lambda 0.0001 --problem "BL-TR"
python3.11  /home/amirrj/projects/def-lelis/amirrj/Sparse-Policies/Sparse-Policies/code/scripts/train_ppo.py \
--env_id Karel_stair_climber \
--seed $SLURM_ARRAY_TASK_ID \
--game_width 12 \
--game_height 12 \
--max_steps 100 \
--num_steps 300 \
--sparse_reward \
--crash_penalty -1.0 \
--karel_seed 0 \
--hidden_size 64 \
--total_timesteps 1500000 \
--num_envs 8 \
--learning_rate 0.001 \
--clip_coef 0.2 \
--ent_coef 0.03 \
--ppo_type gru \
--l1_lambda 0.0001 \
--value_learning_rate 0.05 \
--weight_decay 0.0 \
--exp_name stairClimber_PPO_GRU_sparse_h64_lr0.001_clip0.2_ent0.03_vlr0.05_ks0