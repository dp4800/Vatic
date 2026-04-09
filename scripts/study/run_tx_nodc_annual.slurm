#!/bin/bash
#SBATCH --job-name=tx-nodc-ann
#SBATCH --array=0-3
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/dp4800/Documents/GitHub/Vatic/outputs/TX_2018_ANNUAL/logs/nodc_%a_%j.out
#SBATCH --error=/home/dp4800/Documents/GitHub/Vatic/outputs/TX_2018_ANNUAL/logs/nodc_%a_%j.err

# No-DC baseline simulations: plain Texas-7k (without data center load)
# for 4 representative weeks (one per season) across 2018.
# Outputs go into {week}/no-dc/ alongside existing baseline/ dirs.

REPO=/home/dp4800/Documents/GitHub/Vatic
cd "$REPO"

module purge
module load gurobi/10.0.1
module load anaconda3/2024.10
export PATH="$HOME/.local/bin:$PATH"

# One representative week per season: Winter, Spring, Summer, Fall
DATES=(
    2018-01-07
    2018-04-01
    2018-07-15
    2018-10-21
)

DATE="${DATES[$SLURM_ARRAY_TASK_ID]}"
OUT="outputs/TX_2018_ANNUAL/${DATE}/no-dc"
mkdir -p "$OUT"

echo "============================================================"
echo "  Grid       : Texas-7k (no DC)"
echo "  Date       : $DATE"
echo "  Array task : ${SLURM_ARRAY_TASK_ID}"
echo "  Output     : $OUT"
echo "  Start      : $(date)"
echo "============================================================"

vatic-det Texas-7k "$DATE" 7 \
    --solver gurobi \
    --solver-args Cuts=1 Presolve=1 Heuristics=0.03 \
    --threads 8 \
    --output-detail 2 \
    --ruc-horizon 48 \
    --sced-horizon 4 \
    --ruc-mipgap 0.01 \
    --reserve-factor 0.05 \
    --csv --lmps \
    --out-dir "$OUT"

echo "============================================================"
echo "  Done : $(date)"
echo "============================================================"
