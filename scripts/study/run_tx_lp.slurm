#!/bin/bash
#SBATCH --job-name=tx-lp
#SBATCH --array=0-1
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/dp4800/Documents/GitHub/Vatic/outputs/TX_REAL/logs/tx_lp_%a_%j.out
#SBATCH --error=/home/dp4800/Documents/GitHub/Vatic/outputs/TX_REAL/logs/tx_lp_%a_%j.err

REPO=/home/dp4800/Documents/GitHub/Vatic
cd "$REPO"

mkdir -p outputs/TX_REAL/logs

module purge
module load gurobi/10.0.1
module load anaconda3/2024.10
export PATH="$HOME/.local/bin:$PATH"

CONFIGS=(
    "scripts/study/configs/tx_real_winter.json"
    "scripts/study/configs/tx_real_summer.json"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "============================================================"
echo "  Array task : ${SLURM_ARRAY_TASK_ID}"
echo "  Config     : $CONFIG"
echo "  Start      : $(date)"
echo "  (LP-CAS requeue — steps 1-7 will be skipped)"
echo "============================================================"

python -u scripts/main.py --config "$CONFIG"

echo "============================================================"
echo "  Done : $(date)"
echo "============================================================"
