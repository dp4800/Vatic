#!/bin/bash
#SBATCH --job-name=tx2018-ann
#SBATCH --array=0-23
#SBATCH --time=24:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/dp4800/Documents/GitHub/Vatic/outputs/TX_2018_ANNUAL/logs/tx_2018_%a_%j.out
#SBATCH --error=/home/dp4800/Documents/GitHub/Vatic/outputs/TX_2018_ANNUAL/logs/tx_2018_%a_%j.err

REPO=/home/dp4800/Documents/GitHub/Vatic
cd "$REPO"

mkdir -p outputs/TX_2018_ANNUAL/logs

module purge
module load gurobi/10.0.1
module load anaconda3/2024.10
export PATH="$HOME/.local/bin:$PATH"

CONFIGS=(
    "scripts/study/configs/tx_2018_week00_2018-01-07.json"
    "scripts/study/configs/tx_2018_week01_2018-01-21.json"
    "scripts/study/configs/tx_2018_week02_2018-02-04.json"
    "scripts/study/configs/tx_2018_week03_2018-02-18.json"
    "scripts/study/configs/tx_2018_week04_2018-03-04.json"
    "scripts/study/configs/tx_2018_week05_2018-03-18.json"
    "scripts/study/configs/tx_2018_week06_2018-04-01.json"
    "scripts/study/configs/tx_2018_week07_2018-04-15.json"
    "scripts/study/configs/tx_2018_week08_2018-05-06.json"
    "scripts/study/configs/tx_2018_week09_2018-05-20.json"
    "scripts/study/configs/tx_2018_week10_2018-06-03.json"
    "scripts/study/configs/tx_2018_week11_2018-06-17.json"
    "scripts/study/configs/tx_2018_week12_2018-07-01.json"
    "scripts/study/configs/tx_2018_week13_2018-07-15.json"
    "scripts/study/configs/tx_2018_week14_2018-08-05.json"
    "scripts/study/configs/tx_2018_week15_2018-08-19.json"
    "scripts/study/configs/tx_2018_week16_2018-09-02.json"
    "scripts/study/configs/tx_2018_week17_2018-09-16.json"
    "scripts/study/configs/tx_2018_week18_2018-10-07.json"
    "scripts/study/configs/tx_2018_week19_2018-10-21.json"
    "scripts/study/configs/tx_2018_week20_2018-11-04.json"
    "scripts/study/configs/tx_2018_week21_2018-11-18.json"
    "scripts/study/configs/tx_2018_week22_2018-12-02.json"
    "scripts/study/configs/tx_2018_week23_2018-12-16.json"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "============================================================"
echo "  Grid       : Texas-7k-DC-REAL (2018)"
echo "  Array task : ${SLURM_ARRAY_TASK_ID}"
echo "  Config     : $CONFIG"
echo "  Start      : $(date)"
echo "============================================================"

python -u scripts/main.py --config "$CONFIG"

echo "============================================================"
echo "  Done : $(date)"
echo "============================================================"
