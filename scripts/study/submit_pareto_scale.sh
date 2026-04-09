#!/bin/bash
# submit_pareto_scale.sh
# Submit the full scaling-based Pareto study: 38 jobs total, all queued at once.
#
# 19 portfolios × 2 modes (CAS + no-CAS) = 38 SLURM array tasks
# All run in parallel — well within the ~49-job cluster limit.
# Wall time determined by slowest job (~3 h for high-scale summer scenarios).
#
# Usage:
#   cd /home/dp4800/Documents/GitHub/Vatic
#   bash scripts/study/submit_pareto_scale.sh

set -e

REPO=/home/dp4800/Documents/GitHub/Vatic
STUDY=$REPO/scripts/study

echo "=== Pareto Scale Study — job submission ==="
echo "Repo : $REPO"
echo "Date : $(date)"
echo "Jobs : 19 portfolios × 2 modes = 38 total"
echo

CAS_JID=$(sbatch --parsable "$STUDY/run_scale_cas.slurm")
echo "Submitted CAS   : job $CAS_JID  (array 0-18)"

NOCAS_JID=$(sbatch --parsable "$STUDY/run_scale_nocas.slurm")
echo "Submitted NoCAS : job $NOCAS_JID  (array 0-18)"

echo
echo "Monitor:"
echo "  squeue -u dp4800"
echo "  squeue -j $CAS_JID,$NOCAS_JID"
echo
echo "Logs    : outputs/TX_PARETO_SCALE/logs/"
echo "Results : outputs/TX_PARETO_SCALE/results{,_nocas}/"
