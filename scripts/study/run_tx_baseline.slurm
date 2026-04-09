#!/bin/bash
#SBATCH --job-name=tx-baseline
#SBATCH --array=0-3
#SBATCH --time=22:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu
#SBATCH --output=/home/dp4800/Documents/GitHub/Vatic/outputs/CAS_STUDY_TX_2018/logs/baseline_%a_%j.out
#SBATCH --error=/home/dp4800/Documents/GitHub/Vatic/outputs/CAS_STUDY_TX_2018/logs/baseline_%a_%j.err

REPO=/home/dp4800/Documents/GitHub/Vatic
cd "$REPO"

mkdir -p outputs/CAS_STUDY_TX_2018/logs

module purge
module load gurobi/10.0.1
module load anaconda3/2024.10
export PATH="$HOME/.local/bin:$PATH"

CONFIG=$(python3 -c "
import csv
with open('scripts/study/tx_manifest.csv') as f:
    rows = [r for r in csv.DictReader(f) if r['phase'] == 'baseline']
print(rows[${SLURM_ARRAY_TASK_ID}]['config_path'])
")

LABEL=$(python3 -c "
import csv
with open('scripts/study/tx_manifest.csv') as f:
    rows = [r for r in csv.DictReader(f) if r['phase'] == 'baseline']
print(rows[${SLURM_ARRAY_TASK_ID}]['run_label'])
")

echo "============================================================"
echo "  Array task : ${SLURM_ARRAY_TASK_ID}"
echo "  Label      : $LABEL"
echo "  Config     : $CONFIG"
echo "  Start      : $(date)"
echo "============================================================"

python -u scripts/main.py --config "$CONFIG"

echo "============================================================"
echo "  Done : $(date)"
echo "============================================================"
