#!/bin/bash
# Runs all paper calculations one after another. On a cluster, submit them
# as separate jobs instead (see submit.slurm).
set -e
cd "$(dirname "$0")"
(cd 06_numerics && python run_numerics.py)
(cd 04_h2_adapt_vqe && python run_h2.py --scan)          # separate Qiskit environment
(cd 01_water && python run_water.py)
(cd 03_nitromethane && python run_nitromethane.py)
(cd 05_active_learning && python run_falcon.py --system water)
(cd 05_active_learning && python run_falcon.py --system methane)
(cd 02_ethylene && python run_ethylene.py)               # the expensive one
