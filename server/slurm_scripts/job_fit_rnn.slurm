#!/bin/bash
# Array job: one (hidden_size, session) per task, see `python ../fit_rnn.py --list`.
# 60 tasks = N48 + N6 x 30 sessions. 6 CPUs/task: 5 CV folds + full-data fit in parallel.
# Peak task memory (N48, server) ~ 1.85 GB + 43 MB per 1k trials (BGM3 ~660k -> ~30 GB);
# submit per-size buckets with --mem overriding the 10G default below.
# Submit from server/slurm_scripts/:
#   sbatch --array=0-59 job_fit_rnn.slurm
#   sbatch --array=0-59 job_fit_rnn.slurm --skip-existing   # resume
#SBATCH --job-name=VRNN
#SBATCH --cpus-per-task=6
#SBATCH --mem=10G
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-59
#SBATCH --output=/mnt/pve/Homes/bapun/Data/logs/%x_%A_%a.out

# --- Timestamp for logs ---
basefolder=/mnt/pve/Homes/bapun/Data/
timestamp=$(date +"%Y%m%d_%H%M%S")
logsfolder=${basefolder}/logs
mkdir -p ${logsfolder}

#--- logs cleanup -----
find /mnt/pve/Homes/bapun/Data/logs/ -type f -name "*.out" -mtime +3 -delete
find /mnt/pve/Homes/bapun/Data/logs/ -type f -name "*.err" -mtime +3 -delete

echo "Redirecting detailed stream log to: ${logsfolder}/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${timestamp}.stream.log"
exec > ${logsfolder}/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${timestamp}.stream.log
exec 2>&1

# --- Log job start ---
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "SLURM Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

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

# --- Run one task; extra sbatch args (e.g. --skip-existing) pass through ---
python ../fit_rnn.py --n-jobs-inner 5 "$@"

# --- Log job end ---
echo "Job finished at: $(date)"
