#!/bin/bash

architectures=(
    "test_cartpoleEasy_sweep1_noFE"
    "test_cartpoleEasy_sweep2_FE100_SF0"
    "test_cartpoleEasy_sweep3_FE256_SF0"
    "test_cartpoleEasy_sweep4_FE512_SF0"
    "test_cartpoleEasy_sweep5_FE100_SF50"
    "test_cartpoleEasy_sweep6_FE256_SF50"
    "test_cartpoleEasy_sweep7_FE512_SF50"
)

for arch in "${architectures[@]}"; do
    sbatch --export=ARCH="$arch" run_sweep_parallel.sh
done