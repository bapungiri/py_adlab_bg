#!/bin/bash
# Submit a model-recovery run as a SLURM array (one simulated subject per task:
# n_models x n_subjects tasks) plus a merge job that runs only after every task succeeded.
#
# usage: ./submit_model_recovery_array.sh "<models>" [n_subjects=100] [n_starts=12] [cpus=6] [mem=1200M]
#   e.g. ./submit_model_recovery_array.sh "base sticky decay sticky_decay"
#   models: base sticky decay sticky_decay cost sticky_cost (see model_recovery.MODELS)
#   mem: Qlearn-family fits peak at ~840M MaxRSS; check sacct before raising it.
# Optional environment: EXTRA_ARGS passed to model_recovery.py,
#   e.g. EXTRA_ARGS='--blocks-per-session 5 --tag window5'
set -e
MODELS="$1"
N_SUBJECTS="${2:-100}"; N_STARTS="${3:-12}"; CPUS="${4:-6}"; MEM="${5:-1200M}"
[ -n "$MODELS" ] || { echo "usage: $0 \"<models>\" [n_subjects] [n_starts] [cpus] [mem]"; exit 1; }
cd "$(dirname "$0")"
EXTRA_ARGS="${EXTRA_ARGS:-}"
export MODELS N_SUBJECTS N_STARTS EXTRA_ARGS
N_MODELS=$(echo $MODELS | wc -w)
RANGE="0-$((N_MODELS * N_SUBJECTS - 1))"
NAME="ModelRec_$(echo $MODELS | tr ' ' '_')"
AID=$(sbatch --parsable --array="$RANGE" --cpus-per-task="$CPUS" --mem="$MEM" \
      --job-name="$NAME" job_model_recovery_array.slurm)
MID=$(sbatch --parsable --dependency=afterok:"$AID" --job-name="Merge_${NAME}" \
      job_model_recovery_merge.slurm "$AID")
echo "$NAME: $N_MODELS models x $N_SUBJECTS subjects -> array job $AID ($RANGE, $CPUS CPUs, $MEM each), merge job $MID"
