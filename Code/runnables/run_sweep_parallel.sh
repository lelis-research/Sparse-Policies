#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=07-00:00
#SBATCH --output=job_logs/sweepOrig1/%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-lelis
#SBATCH --mail-user=arajabpo@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-80


source ~/Sparse-Policies/venv/bin/activate

module load flexiblas
export FLEXIBLAS=imkl


learning_rates=(1e-5 1e-4 1e-3)
clip_coefs=(0.05 0.1 0.2)
ent_coefs=(0.1 0.2 0.25)
l1_lambdas=(0.0001 0.005 0.001)

num_lr=${#learning_rates[@]}       # 3
num_clip=${#clip_coefs[@]}         # 3
num_ent=${#ent_coefs[@]}           # 3
num_l1=${#l1_lambdas[@]}           # 3

# total combinations = 3 * 3 * 3 * 3 = 81
idx=$SLURM_ARRAY_TASK_ID

# 1) Get index for L1
l1_index=$(( idx % num_l1 ))            # remainder
idx=$(( idx / num_l1 ))                 # integer division

# 2) Get index for ent_coef
ent_index=$(( idx % num_ent ))
idx=$(( idx / num_ent ))

# 3) Get index for clip_coef
clip_index=$(( idx % num_clip ))
idx=$(( idx / num_clip ))

# 4) Get index for learning_rate
lr_index=$(( idx % num_lr ))

# Extract the actual values
LR="${learning_rates[${lr_index}]}"
CLIP="${clip_coefs[${clip_index}]}"
ENT="${ent_coefs[${ent_index}]}"
L1="${l1_lambdas[${l1_index}]}"


# Run the training script
python ~/Sparse-Policies/Code/scripts/train_ppo.py \
  --env_id Karel_stair_climber \
  --seed 0 \
  --game_width 12 \
  --game_height 12 \
  --max_steps 50 \
  --num_steps 300 \
  --sparse_reward \
  --hidden_size 32 \
  --total_timesteps 2_000_000 \
  --num_envs 1 \
  --num_minibatches 1 \
  --multi_initial_confs \
  --ppo_type original \
  --learning_rate "${LR}" \
  --l1_lambda "${L1}" \
  --ent_coef "${ENT}" \
  --clip_coef "${CLIP}" \
  --exp_name "stair_LR${LR}_L1${L1}_ENT${ENT}_CLIP${CLIP}_sweepOrig1"