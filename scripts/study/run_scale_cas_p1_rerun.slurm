#!/bin/bash
#SBATCH --job-name=tx-scale-cas-rerun
#SBATCH --array=0-18
#SBATCH --time=8:00:00
#SBATCH --qos=short
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/dp4800/Documents/GitHub/Vatic/outputs/TX_PARETO_SCALE/logs/cas_rerun_%a_%j.out
#SBATCH --error=/home/dp4800/Documents/GitHub/Vatic/outputs/TX_PARETO_SCALE/logs/cas_rerun_%a_%j.err

REPO=/home/dp4800/Documents/GitHub/Vatic
cd "$REPO"

mkdir -p outputs/TX_PARETO_SCALE/logs

module purge
module load gurobi/10.0.1
module load anaconda3/2024.10
export PATH="$HOME/.local/bin:$PATH"

set -e

echo "============================================================"
echo "  Scale CAS RERUN (flex_ratio=0.30)  row=${SLURM_ARRAY_TASK_ID}"
echo "  Start : $(date)"
echo "============================================================"

python -u scripts/pareto_eval_scale.py \
    --row "${SLURM_ARRAY_TASK_ID}" \
    --portfolios scripts/study/pareto_portfolios_scale.csv \
    --config scripts/study/params_pareto_scale.json

echo "============================================================"
echo "  Done : $(date)"
echo "============================================================"
