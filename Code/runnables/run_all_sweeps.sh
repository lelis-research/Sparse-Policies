#!/bin/bash
#SBATCH --account=rrg-lelis

architectures=(
    "Quad_sweep1_noFE"
    "Quad_sweep2_FE100_SF0"
    "Quad_sweep3_FE256_SF0"
    "Quad_sweep4_FE512_SF0"
    "Quad_sweep5_FE100_SF50"
    "Quad_sweep6_FE256_SF50"
    "Quad_sweep7_FE512_SF50"
)

for arch in "${architectures[@]}"; do
    sbatch --export=ARCH="$arch" --output="job_logs/${arch}/%N-%j.out" run_sweep_parallel.sh
done