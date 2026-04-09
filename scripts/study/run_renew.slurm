#!/bin/bash
#SBATCH --job-name=cas-renew
#SBATCH --array=0-33
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/dp4800/Documents/GitHub/Vatic/outputs/RENEW_STUDY_RTS_2020/logs/renew_%a_%j.out
#SBATCH --error=/home/dp4800/Documents/GitHub/Vatic/outputs/RENEW_STUDY_RTS_2020/logs/renew_%a_%j.err

REPO=/home/dp4800/Documents/GitHub/Vatic
cd "$REPO"

mkdir -p outputs/RENEW_STUDY_RTS_2020/logs

module purge
module load gurobi/10.0.1
module load anaconda3/2024.10
export PATH="$HOME/.local/bin:$PATH"

# Read manifest row for this array task
CONFIG=$(python3 -c "
import pandas as pd
m = pd.read_csv('scripts/study/renew_manifest.csv')
row = m[m['run_id'] == ${SLURM_ARRAY_TASK_ID}].iloc[0]
print(row['config_path'])
")

LABEL=$(python3 -c "
import pandas as pd
m = pd.read_csv('scripts/study/renew_manifest.csv')
row = m[m['run_id'] == ${SLURM_ARRAY_TASK_ID}].iloc[0]
print(row['run_label'])
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
