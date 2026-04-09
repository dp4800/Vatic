#!/bin/bash
#SBATCH --job-name=tx2018-axd
#SBATCH --array=0-127
#SBATCH --time=1-12:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/dp4800/Documents/GitHub/Vatic/outputs/TX_2018_ANNUAL/logs/tx_axd_%a_%j.out
#SBATCH --error=/home/dp4800/Documents/GitHub/Vatic/outputs/TX_2018_ANNUAL/logs/tx_axd_%a_%j.err

REPO=/home/dp4800/Documents/GitHub/Vatic
cd "$REPO"

mkdir -p outputs/TX_2018_ANNUAL/logs

module purge
module load gurobi/10.0.1
module load anaconda3/2024.10
export PATH="$HOME/.local/bin:$PATH"

mapfile -t CONFIGS < scripts/study/configs/tx_2018_axd_list.txt

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "============================================================"
echo "  Grid       : tx_2018 alpha×deferral"
echo "  Array task : ${SLURM_ARRAY_TASK_ID}"
echo "  Config     : $CONFIG"
echo "  Start      : $(date)"
echo "============================================================"

python -u scripts/main.py --config "$CONFIG"

echo "============================================================"
echo "  Done : $(date)"
echo "============================================================"
