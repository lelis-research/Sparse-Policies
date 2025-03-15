#!/bin/bash
#SBATCH --account=rrg-lelis

architectures=(
    "cartpoleEasy_sweep1_noFE"
    "cartpoleEasy_sweep2_FE100_SF0"
    "cartpoleEasy_sweep3_FE256_SF0"
    "cartpoleEasy_sweep4_FE512_SF0"
    "cartpoleEasy_sweep5_FE100_SF50"
    "cartpoleEasy_sweep6_FE256_SF50"
    "cartpoleEasy_sweep7_FE512_SF50"
)

for arch in "${architectures[@]}"; do
    sbatch --export=ARCH="$arch" --output="job_logs/${arch}/%N-%j.out" run_sweep_parallel.sh
done