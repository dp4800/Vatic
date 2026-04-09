#!/bin/bash
#SBATCH --job-name=tx2018-247fix
#SBATCH --array=0-10
#SBATCH --time=24:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/dp4800/Documents/GitHub/Vatic/outputs/TX_2018_ANNUAL/logs/sim247fix_%a_%j.out
#SBATCH --error=/home/dp4800/Documents/GitHub/Vatic/outputs/TX_2018_ANNUAL/logs/sim247fix_%a_%j.err

REPO=/home/dp4800/Documents/GitHub/Vatic
cd "$REPO"

mkdir -p outputs/TX_2018_ANNUAL/logs

module purge
module load gurobi/10.0.1
module load anaconda3/2024.10
export PATH="$HOME/.local/bin:$PATH"

CONFIGS=(
    "scripts/study/configs/tx_2018_week00_2018-01-07.json"
    "scripts/study/configs/tx_2018_week03_2018-02-18.json"
    "scripts/study/configs/tx_2018_week04_2018-03-04.json"
    "scripts/study/configs/tx_2018_week05_2018-03-18.json"
    "scripts/study/configs/tx_2018_week08_2018-05-06.json"
    "scripts/study/configs/tx_2018_week10_2018-06-03.json"
    "scripts/study/configs/tx_2018_week16_2018-09-02.json"
    "scripts/study/configs/tx_2018_week20_2018-11-04.json"
    "scripts/study/configs/tx_2018_week21_2018-11-18.json"
    "scripts/study/configs/tx_2018_week23_2018-12-16.json"
    "scripts/study/configs/tx_2018_week15_2018-08-19.json"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "============================================================"
echo "  Grid       : Texas-7k-DC-REAL (2018) - sim-247 fix"
echo "  Array task : ${SLURM_ARRAY_TASK_ID}"
echo "  Config     : $CONFIG"
echo "  Start      : $(date)"
echo "============================================================"

python -u scripts/main.py --config "$CONFIG"

echo "============================================================"
echo "  Done : $(date)"
echo "============================================================"
