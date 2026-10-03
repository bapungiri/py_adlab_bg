#!/bin/bash
# Submit a fit preset as a SLURM array (one session per task) plus a merge job
# that runs only after every task succeeded.
#
# usage: ./submit_fit_array.sh <preset> [cpus_per_task=6] [mem=4G] [max_concurrent]
#   e.g. ./submit_fit_array.sh sticky_p8020_intact
#        ./submit_fit_array.sh sticky_p9505 5 4G 4
set -e
PRESET="$1"; CPUS="${2:-6}"; MEM="${3:-4G}"; THROTTLE="${4:-}"
[ -n "$PRESET" ] || { echo "usage: $0 <preset> [cpus] [mem] [max_concurrent]"; exit 1; }
cd "$(dirname "$0")"

eval "$(/mnt/pve/Homes/conda/miniconda3/bin/conda shell.bash hook)"
conda activate bapun_conda_env
export PYTHONPATH=/mnt/pve/Homes/bapun/Codes/NeuroPy:/mnt/pve/Homes/bapun/Codes/BanditPy:/mnt/pve/Homes/bapun/Codes/py_adlab_bg:$PYTHONPATH

N=$(cd .. && python fit_policy_wrapper.py --preset "$PRESET" --count | tail -1)
[[ "$N" =~ ^[0-9]+$ ]] && [ "$N" -gt 0 ] || { echo "could not count sessions for $PRESET: $N"; exit 1; }
RANGE="0-$((N - 1))${THROTTLE:+%$THROTTLE}"

AID=$(sbatch --parsable --array="$RANGE" --cpus-per-task="$CPUS" --mem="$MEM" \
      --job-name="Fit_${PRESET}" job_fit_policy_array.slurm "$PRESET")
MID=$(sbatch --parsable --dependency=afterok:"$AID" --job-name="Merge_${PRESET}" \
      job_fit_policy_merge.slurm "$PRESET" "$AID")
echo "$PRESET: $N sessions -> array job $AID ($RANGE, $CPUS CPUs, $MEM each), merge job $MID"
