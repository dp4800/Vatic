#!/bin/bash
#SBATCH --job-name=tx-sensitivity
#SBATCH --array=0-23
#SBATCH --time=22:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu
#SBATCH --output=/home/dp4800/Documents/GitHub/Vatic/outputs/CAS_STUDY_TX_2018/logs/sens_%a_%j.out
#SBATCH --error=/home/dp4800/Documents/GitHub/Vatic/outputs/CAS_STUDY_TX_2018/logs/sens_%a_%j.err

# NOTE: The Texas-7k-DC-SCALED grid must be pre-built before this array runs.
# The load_growth variants (tasks 0 and 12) use it; if the grid does not exist,
# those tasks will create it on first run — but only ONE task should do so.
# Safe approach: run tasks 0 and 12 first (array=0-0 then 12-12), then submit
# the full array after the SCALED grid exists.

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
    rows = [r for r in csv.DictReader(f) if r['phase'] == 'sensitivity']
print(rows[${SLURM_ARRAY_TASK_ID}]['config_path'])
")

GROUP=$(python3 -c "
import csv
with open('scripts/study/tx_manifest.csv') as f:
    rows = [r for r in csv.DictReader(f) if r['phase'] == 'sensitivity']
print(rows[${SLURM_ARRAY_TASK_ID}]['group'])
")

LABEL=$(python3 -c "
import csv
with open('scripts/study/tx_manifest.csv') as f:
    rows = [r for r in csv.DictReader(f) if r['phase'] == 'sensitivity']
print(rows[${SLURM_ARRAY_TASK_ID}]['run_label'])
")

echo "============================================================"
echo "  Array task : ${SLURM_ARRAY_TASK_ID}"
echo "  Group      : $GROUP"
echo "  Label      : $LABEL"
echo "  Config     : $CONFIG"
echo "  Start      : $(date)"
echo "============================================================"

python -u scripts/main.py --config "$CONFIG"

echo "============================================================"
echo "  Done : $(date)"
echo "============================================================"
