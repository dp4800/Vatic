#!/bin/bash
#SBATCH --job-name=tx-renew-W10000
#SBATCH --array=0-23
#SBATCH --time=24:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/dp4800/Documents/GitHub/Vatic/outputs/TX_RENEW_W10000/logs/tx_renew_%a_%j.out
#SBATCH --error=/home/dp4800/Documents/GitHub/Vatic/outputs/TX_RENEW_W10000/logs/tx_renew_%a_%j.err

REPO=/home/dp4800/Documents/GitHub/Vatic
cd "$REPO"

mkdir -p outputs/TX_RENEW_W10000/logs

module purge
module load gurobi/10.0.1
module load anaconda3/2024.10
export PATH="$HOME/.local/bin:$PATH"

CONFIGS=(
    "tx_renew_W10000_week00_2018-01-07.json"
    "tx_renew_W10000_week01_2018-01-21.json"
    "tx_renew_W10000_week02_2018-02-04.json"
    "tx_renew_W10000_week03_2018-02-18.json"
    "tx_renew_W10000_week04_2018-03-04.json"
    "tx_renew_W10000_week05_2018-03-18.json"
    "tx_renew_W10000_week06_2018-04-01.json"
    "tx_renew_W10000_week07_2018-04-15.json"
    "tx_renew_W10000_week08_2018-05-06.json"
    "tx_renew_W10000_week09_2018-05-20.json"
    "tx_renew_W10000_week10_2018-06-03.json"
    "tx_renew_W10000_week11_2018-06-17.json"
    "tx_renew_W10000_week12_2018-07-01.json"
    "tx_renew_W10000_week13_2018-07-15.json"
    "tx_renew_W10000_week14_2018-08-05.json"
    "tx_renew_W10000_week15_2018-08-19.json"
    "tx_renew_W10000_week16_2018-09-02.json"
    "tx_renew_W10000_week17_2018-09-16.json"
    "tx_renew_W10000_week18_2018-10-07.json"
    "tx_renew_W10000_week19_2018-10-21.json"
    "tx_renew_W10000_week20_2018-11-04.json"
    "tx_renew_W10000_week21_2018-11-18.json"
    "tx_renew_W10000_week22_2018-12-02.json"
    "tx_renew_W10000_week23_2018-12-16.json"
)

CONFIG="scripts/study/configs/${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "================================================================"
echo "  Scenario    : W10000"
echo "  Array task  : ${SLURM_ARRAY_TASK_ID}"
echo "  Config      : $CONFIG"
echo "  Start       : $(date)"
echo "================================================================"

python -u scripts/main.py --config "$CONFIG"

echo "================================================================"
echo "  Done : $(date)"
echo "================================================================"
