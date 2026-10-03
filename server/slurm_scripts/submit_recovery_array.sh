#!/bin/bash
# Submit a parameter-recovery run as a SLURM array (one simulated subject per
# task) plus a merge job that runs only after every task succeeded.
#
# usage: ./submit_recovery_array.sh <design> <variant> [optimizer=lbfgs] [n_starts=12]
#                                   [n_subjects=100] [cpus=6] [mem=2G] [max_concurrent]
#   e.g. ./submit_recovery_array.sh struc_unstruc_tier sticky
# Optional environment: POLICY (default Qlearn), EXTRA_ARGS passed to param_recovery.py,
#   e.g. EXTRA_ARGS='--true-range sticky=0,5 --tag truesticky0to5'
set -e
DESIGN="$1"; VARIANT="$2"
OPTIMIZER="${3:-lbfgs}"; N_STARTS="${4:-12}"; MAX_SUBJECTS="${5:-100}"
CPUS="${6:-6}"; MEM="${7:-2G}"; THROTTLE="${8:-}"
[ -n "$DESIGN" ] && [ -n "$VARIANT" ] || { echo "usage: $0 <design> <variant> [optimizer] [n_starts] [n_subjects] [cpus] [mem] [max_concurrent]"; exit 1; }
cd "$(dirname "$0")"
POLICY="${POLICY:-Qlearn}"; EXTRA_ARGS="${EXTRA_ARGS:-}"
export DESIGN VARIANT OPTIMIZER N_STARTS MAX_SUBJECTS POLICY EXTRA_ARGS
RANGE="0-$((MAX_SUBJECTS - 1))${THROTTLE:+%$THROTTLE}"
NAME="Rec_${POLICY}_${DESIGN}_${VARIANT}"
AID=$(sbatch --parsable --array="$RANGE" --cpus-per-task="$CPUS" --mem="$MEM" \
      --job-name="$NAME" job_param_recovery_array.slurm)
MID=$(sbatch --parsable --dependency=afterok:"$AID" --job-name="Merge_${NAME}" \
      job_param_recovery_merge.slurm "$AID")
echo "$NAME: $MAX_SUBJECTS subjects -> array job $AID ($RANGE, $CPUS CPUs, $MEM each), merge job $MID"
