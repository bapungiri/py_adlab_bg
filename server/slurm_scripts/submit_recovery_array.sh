#!/bin/bash
# Submit a parameter-recovery run as a SLURM array (one simulated subject per
# task) plus a merge job that runs only after every task succeeded.
#
# usage: ./submit_recovery_array.sh <design> <variant> [optimizer=lbfgs] [n_starts=10]
#                                   [n_subjects=100] [cpus=10] [mem=2G] [max_concurrent]
#   e.g. ./submit_recovery_array.sh struc_unstruc_tier sticky
set -e
DESIGN="$1"; VARIANT="$2"
OPTIMIZER="${3:-lbfgs}"; N_STARTS="${4:-10}"; MAX_SUBJECTS="${5:-100}"
CPUS="${6:-10}"; MEM="${7:-2G}"; THROTTLE="${8:-}"
[ -n "$DESIGN" ] && [ -n "$VARIANT" ] || { echo "usage: $0 <design> <variant> [optimizer] [n_starts] [n_subjects] [cpus] [mem] [max_concurrent]"; exit 1; }
cd "$(dirname "$0")"
export DESIGN VARIANT OPTIMIZER N_STARTS MAX_SUBJECTS
RANGE="0-$((MAX_SUBJECTS - 1))${THROTTLE:+%$THROTTLE}"
NAME="Rec_${DESIGN}_${VARIANT}"
AID=$(sbatch --parsable --array="$RANGE" --cpus-per-task="$CPUS" --mem="$MEM" \
      --job-name="$NAME" job_param_recovery_array.slurm)
MID=$(sbatch --parsable --dependency=afterok:"$AID" --job-name="Merge_${NAME}" \
      job_param_recovery_merge.slurm "$AID")
echo "$NAME: $MAX_SUBJECTS subjects -> array job $AID ($RANGE, $CPUS CPUs, $MEM each), merge job $MID"
