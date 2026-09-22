#!/bin/bash
#SBATCH --job-name=PRQlearn
#SBATCH --cpus-per-task=20        # 20 outer jobs * 5 inner jobs
#SBATCH --mem=96G                  # leave ~32G RAM free for others
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# --- Job parallel config ---
N_JOBS_SUBJECT=20
N_JOBS_INNER=5

# --- Resolve paths relative to this script and set requested CPU count ---
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# DATA_DIR="$(cd "${PROJECT_ROOT}/../Data" && pwd)"

SCRIPT_DIR=/mnt/pve/Homes/bapun/Codes/py_adlab_bg/server/slurm_scripts
PROJECT_ROOT=/mnt/pve/Homes/bapun/Codes/py_adlab_bg/server
DATA_DIR=/mnt/pve/Homes/bapun/Data

# --- Timestamp for logs ---
basefolder=${DATA_DIR}
timestamp=$(date +"%Y%m%d_%H%M%S")
logsfolder=${basefolder}/logs
mkdir -p ${logsfolder}

#--- logs cleanup -----
find ${DATA_DIR}/logs/ -type f -name "*.out" -mtime +3 -delete
find ${DATA_DIR}/logs/ -type f -name "*.err" -mtime +3 -delete

echo "Redirecting detailed stream log to: ${logsfolder}/${SLURM_JOB_NAME}_${timestamp}_${SLURM_JOB_ID}.stream.log"
exec > ${logsfolder}/${SLURM_JOB_NAME}_${timestamp}_${SLURM_JOB_ID}.stream.log
exec 2>&1

# --- Log job start ---
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "SLURM Job ID: $SLURM_JOB_ID"

# --- Conda activation ---
eval "$(/mnt/pve/Homes/conda/miniconda3/bin/conda shell.bash hook)"
conda activate bapun_conda_env

# --- Thread / BLAS caps (to avoid oversubscription inside joblib) ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# --- Export custom PYTHONPATH ---
export PYTHONPATH=/mnt/pve/Homes/bapun/Codes/NeuroPy:/mnt/pve/Homes/bapun/Codes/BanditPy:/mnt/pve/Homes/bapun/Codes/py_adlab_bg:$PYTHONPATH

# --- Run recovery with configured parallelism ---
python ../model_recovery.py \
    --max-subjects 100 \
    --n-jobs-subject ${N_JOBS_SUBJECT} \
    --n-jobs-inner ${N_JOBS_INNER}

# --- Log job end ---
echo "Job finished at: $(date)"