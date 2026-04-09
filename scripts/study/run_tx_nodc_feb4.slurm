#!/bin/bash
#SBATCH --job-name=tx-nodc-feb4
#SBATCH --time=28:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/dp4800/Documents/GitHub/Vatic/outputs/TX_NODC_FEB4/logs/run_%j.out
#SBATCH --error=/home/dp4800/Documents/GitHub/Vatic/outputs/TX_NODC_FEB4/logs/run_%j.err

cd /home/dp4800/Documents/GitHub/Vatic
module purge
module load gurobi/10.0.1
module load anaconda3/2024.10
export PATH="$HOME/.local/bin:$PATH"

OUT=outputs/TX_NODC_FEB4/2018-02-04/baseline
mkdir -p "$OUT"

echo "Start: $(date)"
vatic-det Texas-7k 2018-02-04 7 \
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
echo "Done: $(date)"
