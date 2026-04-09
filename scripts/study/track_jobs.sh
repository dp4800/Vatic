#!/bin/bash
# Live job tracker — run on Della login node
# Usage: bash scripts/study/track_jobs.sh [interval_seconds]
#   Default interval: 60s. Ctrl-C to stop.

INTERVAL=${1:-60}
REPO=/home/dp4800/Documents/GitHub/Vatic
OUTDIR="$REPO/outputs/TX_2018_ANNUAL"
PARETO_DIR="$REPO/outputs/TX_PARETO/results"

# ANSI colours
BOLD='\033[1m'
CYAN='\033[1;36m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
DIM='\033[2m'
RST='\033[0m'

bar() {
    local done=$1 total=$2 width=30
    local filled=$(( done * width / (total > 0 ? total : 1) ))
    local empty=$(( width - filled ))
    printf "${GREEN}%${filled}s${RST}${DIM}%${empty}s${RST}" | tr ' ' '█' | tr ' ' '░'
}

pct() { awk "BEGIN{printf \"%.1f\", ($1/$2)*100}" 2>/dev/null; }

count_axd() {
    local n=0
    for d in "$OUTDIR"/*/; do
        [[ "$(basename $d)" =~ ^[0-9]{4} ]] || continue
        for sub in "$d"cas-lp-axd-*/; do
            [ -f "${sub}iter0/cas_results.csv" ] && ((n++))
        done
    done
    echo $n
}

count_axdf() {
    local n=0
    for d in "$OUTDIR"/*/; do
        [[ "$(basename $d)" =~ ^[0-9]{4} ]] || continue
        for sub in "$d"cas-lp-axdf-*/; do
            [ -f "${sub}iter0/cas_results.csv" ] && ((n++))
        done
    done
    echo $n
}

count_sens() {
    local n=0
    for d in "$OUTDIR"/*/; do
        [[ "$(basename $d)" =~ ^[0-9]{4} ]] || continue
        for tag in alpha deferral flex; do
            for sub in "$d"cas-lp-${tag}-*/; do
                [ -f "${sub}iter0/cas_results.csv" ] && ((n++))
            done
        done
    done
    echo $n
}

count_pareto() {
    ls "$PARETO_DIR"/*.json 2>/dev/null | wc -l
}

while true; do
    clear
    NOW=$(date '+%Y-%m-%d %H:%M:%S')

    # ── Slurm queue ──────────────────────────────────────────────────
    QUEUE=$(squeue -u dp4800 --format="%.10i %.20j %.8T %.10M %.5C" 2>/dev/null)
    RUNNING=$(echo "$QUEUE" | grep -c RUNNING 2>/dev/null || echo 0)
    PENDING=$(echo "$QUEUE" | grep -c PENDING 2>/dev/null || echo 0)

    # ── Completion counts ─────────────────────────────────────────────
    SENS_DONE=$(count_sens)
    AXD_DONE=$(count_axd)
    AXDF_DONE=$(count_axdf)
    PARETO_DONE=$(count_pareto)

    SENS_TOT=88
    AXD_TOT=128
    AXDF_TOT=384
    PARETO_TOT=150

    TOTAL_DONE=$(( SENS_DONE + AXD_DONE + AXDF_DONE + PARETO_DONE ))
    TOTAL_TOT=$(( SENS_TOT + AXD_TOT + AXDF_TOT + PARETO_TOT ))

    # ── Print ─────────────────────────────────────────────────────────
    echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════╗${RST}"
    echo -e "${BOLD}${CYAN}║         TX-2018 Sensitivity Study — Live Tracker      ║${RST}"
    echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════╝${RST}"
    echo -e "  ${DIM}Updated: $NOW   (refresh every ${INTERVAL}s — Ctrl-C to stop)${RST}"
    echo ""

    echo -e "${BOLD}  Slurm Queue${RST}"
    echo -e "  Running: ${GREEN}${RUNNING}${RST}   Pending: ${YELLOW}${PENDING}${RST}"
    echo ""

    # Per-sweep rows
    echo -e "${BOLD}  Sweep            Done  /  Total   Progress                  Pct${RST}"
    echo    "  ─────────────────────────────────────────────────────────────────"

    printf  "  %-16s  %4d / %4d   " "OAAT sens"   $SENS_DONE   $SENS_TOT
    bar $SENS_DONE $SENS_TOT
    printf "  %5s%%\n" "$(pct $SENS_DONE $SENS_TOT)"

    printf  "  %-16s  %4d / %4d   " "AXD"         $AXD_DONE    $AXD_TOT
    bar $AXD_DONE $AXD_TOT
    printf "  %5s%%\n" "$(pct $AXD_DONE $AXD_TOT)"

    printf  "  %-16s  %4d / %4d   " "AXDF"        $AXDF_DONE   $AXDF_TOT
    bar $AXDF_DONE $AXDF_TOT
    printf "  %5s%%\n" "$(pct $AXDF_DONE $AXDF_TOT)"

    printf  "  %-16s  %4d / %4d   " "Pareto"      $PARETO_DONE $PARETO_TOT
    bar $PARETO_DONE $PARETO_TOT
    printf "  %5s%%\n" "$(pct $PARETO_DONE $PARETO_TOT)"

    echo    "  ─────────────────────────────────────────────────────────────────"
    printf  "  %-16s  %4d / %4d   " "TOTAL"       $TOTAL_DONE  $TOTAL_TOT
    bar $TOTAL_DONE $TOTAL_TOT
    printf "  %5s%%\n" "$(pct $TOTAL_DONE $TOTAL_TOT)"
    echo ""

    # ── Recent completions ────────────────────────────────────────────
    echo -e "${BOLD}  Recently completed (last 5 AXD/AXDF results):${RST}"
    for d in "$OUTDIR"/*/; do
        [[ "$(basename $d)" =~ ^[0-9]{4} ]] || continue
        for sub in "$d"cas-lp-ax*/; do
            f="${sub}iter0/cas_results.csv"
            [ -f "$f" ] && echo "$f"
        done
    done | xargs -I{} stat --format="%Y {}" {} 2>/dev/null \
      | sort -rn | head -5 \
      | while read ts fp; do
            age=$(( $(date +%s) - ts ))
            if   [ $age -lt 3600 ]; then ago="${age}s ago"
            elif [ $age -lt 86400 ]; then ago="$(( age/3600 ))h ago"
            else ago="$(( age/86400 ))d ago"
            fi
            tag=$(echo "$fp" | grep -oP 'cas-lp-ax[^/]+')
            date=$(echo "$fp" | grep -oP '\d{4}-\d{2}-\d{2}' | head -1)
            printf "    ${DIM}%-14s  %-45s${RST}\n" "$ago" "${date}/${tag}"
        done
    echo ""

    # ── Slurm detail ─────────────────────────────────────────────────
    if [ -n "$(echo "$QUEUE" | tail -n +2)" ]; then
        echo -e "${BOLD}  Active jobs:${RST}"
        echo "$QUEUE" | tail -n +2 | head -10 | while read line; do
            echo "    $line"
        done
    fi

    sleep "$INTERVAL"
done
