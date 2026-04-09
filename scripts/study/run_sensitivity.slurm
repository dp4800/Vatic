#!/bin/bash
#SBATCH --job-name=cas-sensitivity
#SBATCH --array=0-23
#SBATCH --output=/home/dp4800/Documents/GitHub/Vatic/outputs/CAS_STUDY_RTS_2020/logs/sens_%a_%j.out
#SBATCH --error=/home/dp4800/Documents/GitHub/Vatic/outputs/CAS_STUDY_RTS_2020/logs/sens_%a_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu

module purge
module load gurobi/10.0.1
module load anaconda3/2024.10
export PATH=$HOME/.local/bin:$PATH
export -f module

REPO=/home/dp4800/Documents/GitHub/Vatic
MANIFEST=$REPO/scripts/study/manifest.csv

cd $REPO
mkdir -p outputs/CAS_STUDY_RTS_2020/logs

CONFIG=$(python3 -c "
import csv
with open('$MANIFEST') as f:
    rows = [r for r in csv.DictReader(f) if r['phase'] == 'sensitivity']
print(rows[${SLURM_ARRAY_TASK_ID}]['config_path'])
")

GROUP=$(python3 -c "
import csv
with open('$MANIFEST') as f:
    rows = [r for r in csv.DictReader(f) if r['phase'] == 'sensitivity']
print(rows[${SLURM_ARRAY_TASK_ID}]['group'])
")

LABEL=$(python3 -c "
import csv
with open('$MANIFEST') as f:
    rows = [r for r in csv.DictReader(f) if r['phase'] == 'sensitivity']
print(rows[${SLURM_ARRAY_TASK_ID}]['run_label'])
")

echo "============================================================"
echo "  Array task : ${SLURM_ARRAY_TASK_ID}"
echo "  Group      : ${GROUP}"
echo "  Label      : ${LABEL}"
echo "  Config     : ${CONFIG}"
echo "  Start      : $(date)"
echo "============================================================"

python -u scripts/main.py --config "$CONFIG"

echo "============================================================"
echo "  Done : $(date)"
echo "============================================================"
