#!/bin/bash
#SBATCH --job-name=ParamRecQlearn
#SBATCH --cpus-per-task=20        # N_JOBS_SUBJECT * N_JOBS_INNER
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# --- Job parallel config ---
MAX_SUBJECTS=10
N_JOBS_SUBJECT=10
N_JOBS_INNER=2

# --- Resolve paths ---
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

# --- Run param recovery with configured parallelism ---
cd ${PROJECT_ROOT}
python param_recovery.py \
    --policy Qlearn \
    --max-subjects ${MAX_SUBJECTS} \
    --n-jobs-subject ${N_JOBS_SUBJECT} \
    --n-jobs-inner ${N_JOBS_INNER}

# --- Log job end ---
echo "Job finished at: $(date)"
