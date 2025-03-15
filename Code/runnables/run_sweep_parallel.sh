#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00-05:00
#SBATCH --output=job_logs/${ARCH}/%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=rrg-lelis
#SBATCH --mail-user=arajabpo@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0


source ~/scratch/Sparse-Policies/venv/bin/activate

module load flexiblas
export FLEXIBLAS=imkl

seeds=(1 2 3)
learning_rates=(0.001 0.0001 0.00001)
clip_coefs=(0.01 0.1 0.2)
ent_coefs=(0.01 0.1 0.2)
l1_lambdas=(0.0 0.0001 0.0005 0.001)
hiddens=(32 64)

num_seed=${#seeds[@]}              # 3   
num_lr=${#learning_rates[@]}       # 3
num_clip=${#clip_coefs[@]}         # 3
num_ent=${#ent_coefs[@]}           # 3
num_l1=${#l1_lambdas[@]}           # 4
num_hidden=${#hiddens[@]}          # 2

# total combinations = 3*3*3*3*4*2 = 648
idx=$SLURM_ARRAY_TASK_ID

# Get index for hidden size
h_index=$(( idx % num_hidden ))
idx=$(( idx / num_hidden ))

# Get index for L1
l1_index=$(( idx % num_l1 ))            # remainder
idx=$(( idx / num_l1 ))                 # integer division

# Get index for ent_coef
ent_index=$(( idx % num_ent ))
idx=$(( idx / num_ent ))

# Get index for clip_coef
clip_index=$(( idx % num_clip ))
idx=$(( idx / num_clip ))

# Get index for learning_rate
lr_index=$(( idx % num_lr ))
idx=$(( idx / num_lr ))

# Get index for seed
sd_index=$(( idx % num_seed ))


# Extract the actual values
SD="${seeds[${sd_index}]}"
LR="${learning_rates[${lr_index}]}"
CLIP="${clip_coefs[${clip_index}]}"
ENT="${ent_coefs[${ent_index}]}"
L1="${l1_lambdas[${l1_index}]}"
H="${hiddens[${h_index}]}"


# Run the training script
python ~/scratch/Sparse-Policies/Code/scripts/train_ppo.py \
  --env_id CartpoleEasy \
  --seed "${SD}" \
  --num_steps 250 \
  --hidden_size "${H}" \
  --total_timesteps 2_000_000 \
  --num_envs 1 \
  --num_minibatches 1 \
  --ppo_type original \
  --learning_rate "${LR}" \
  --l1_lambda "${L1}" \
  --ent_coef "${ENT}" \
  --clip_coef "${CLIP}" \
  --exp_name "${ARCH}_SD${SD}_LR${LR}_CLIP${CLIP}_ENT${ENT}_L1${L1}_H${H}"