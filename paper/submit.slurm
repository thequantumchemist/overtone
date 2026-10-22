#!/bin/bash
#SBATCH --job-name=overtone
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=7-00:00:00
#SBATCH --output=%x-%j.out
# usage: sbatch submit.slurm 01_water run_water.py [--nprocs 16]
# Adjust partition, modules and environment to your cluster.
module load orca 2>/dev/null || true
export ORCA_COMMAND=${ORCA_COMMAND:-/cluster/applications/ORCA/orca_5_0_4/orca}
cd "$SLURM_SUBMIT_DIR/$1"
shift
python "$@"
