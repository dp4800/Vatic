#!/bin/bash
#SBATCH --job-name=tx-pareto-nocas
#SBATCH --array=0-149
#SBATCH --time=1-12:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/dp4800/Documents/GitHub/Vatic/outputs/TX_PARETO/logs/nocas_%a_%j.out
#SBATCH --error=/home/dp4800/Documents/GitHub/Vatic/outputs/TX_PARETO/logs/nocas_%a_%j.err

REPO=/home/dp4800/Documents/GitHub/Vatic
cd "$REPO"

mkdir -p outputs/TX_PARETO/logs

module purge
module load gurobi/10.0.1
module load anaconda3/2024.10
export PATH="$HOME/.local/bin:$PATH"

set -e

echo "============================================================"
echo "  Pareto no-CAS task : ${SLURM_ARRAY_TASK_ID}"
echo "  Start              : $(date)"
echo "============================================================"

python -u scripts/pareto_eval_nocas.py \
    --row "${SLURM_ARRAY_TASK_ID}" \
    --portfolios scripts/study/pareto_portfolios.csv \
    --config scripts/study/params_pareto.json

echo "============================================================"
echo "  Done : $(date)"
echo "============================================================"
