#!/bin/bash
# submit_pareto_scale_p2.sh
# Phase 2: Add spring (2018-04-15) + fall (2018-10-21) weeks to the
# existing Pareto scale study.  Existing winter+summer outputs untouched.
#
# 19 portfolios × 2 modes (CAS + NoCAS) = 38 SLURM array tasks
# CAS retry (16h) auto-submitted for any tasks that hit the 8h limit.
# NoCAS depends on CAS so signal sims are cached (no duplicate VATIC runs).
#
# Usage:
#   cd /home/dp4800/Documents/GitHub/Vatic
#   bash scripts/study/submit_pareto_scale_p2.sh

set -e

REPO=/home/dp4800/Documents/GitHub/Vatic
STUDY=$REPO/scripts/study

echo "=== Pareto Scale P2 (spring+fall) — job submission ==="
echo "Repo : $REPO"
echo "Date : $(date)"
echo "Jobs : 19 portfolios × 2 modes + retry = up to 57 tasks"
echo

# 1. CAS first (runs signal sims + builds CAS-LP-P2 grids + CAS sims)
CAS_JID=$(sbatch --parsable "$STUDY/run_scale_cas_p2.slurm")
echo "Submitted CAS P2      : job $CAS_JID  (array 0-18, 8h)"

# 2. CAS retry — runs after CAS finishes; completed tasks skip instantly
RETRY_JID=$(sbatch --parsable --dependency=afterany:$CAS_JID "$STUDY/run_scale_cas_p2_retry.slurm")
echo "Submitted CAS P2 retry: job $RETRY_JID  (array 0-18, 16h, after $CAS_JID)"

# 3. NoCAS after CAS — signal sims cached, just extracts inv-only metrics
NOCAS_JID=$(sbatch --parsable --dependency=afterany:$CAS_JID "$STUDY/run_scale_nocas_p2.slurm")
echo "Submitted NoCAS P2    : job $NOCAS_JID  (array 0-18, 8h, after $CAS_JID)"

echo
echo "Monitor:"
echo "  squeue -u dp4800"
echo "  squeue -j $CAS_JID,$RETRY_JID,$NOCAS_JID"
echo
echo "Logs       : outputs/TX_PARETO_SCALE/logs/cas_p2_*.out"
echo "CAS results: outputs/TX_PARETO_SCALE/results_p2/"
echo "NoCAS      : outputs/TX_PARETO_SCALE/results_nocas_p2/"
