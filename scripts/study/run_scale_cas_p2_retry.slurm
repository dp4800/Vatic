#!/bin/bash
#SBATCH --job-name=tx-cas-p2-retry
#SBATCH --array=0-18
#SBATCH --time=16:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/dp4800/Documents/GitHub/Vatic/outputs/TX_PARETO_SCALE/logs/cas_p2_retry_%a_%j.out
#SBATCH --error=/home/dp4800/Documents/GitHub/Vatic/outputs/TX_PARETO_SCALE/logs/cas_p2_retry_%a_%j.err

# Retry for CAS P2 tasks that timed out at 8h.
# Completed tasks (status=ok in result JSON) skip instantly.

REPO=/home/dp4800/Documents/GitHub/Vatic
cd "$REPO"

module purge
module load gurobi/10.0.1
module load anaconda3/2024.10
export PATH="$HOME/.local/bin:$PATH"

set -e

echo "============================================================"
echo "  CAS P2 RETRY  row=${SLURM_ARRAY_TASK_ID}"
echo "  Start : $(date)"
echo "============================================================"

python -u scripts/pareto_eval_scale.py \
    --row "${SLURM_ARRAY_TASK_ID}" \
    --portfolios scripts/study/pareto_portfolios_scale.csv \
    --config scripts/study/params_pareto_scale_p2.json

echo "============================================================"
echo "  Done : $(date)"
echo "============================================================"
