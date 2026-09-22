#!/bin/bash
# One-node run: 15 subjects in parallel, each with 5 cores
#SBATCH --job-name=VRNN15x5
#SBATCH --cpus-per-task=75          # total CPUs used by this job
#SBATCH --mem=80G                  # leave ~32G RAM free for others
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# --- Timestamp for logs ---
basefolder=/mnt/pve/Homes/bapun/Data/
timestamp=$(date +"%Y%m%d_%H%M%S")
logsfolder=${basefolder}/logs
mkdir -p ${logsfolder}

#--- logs cleanup -----
find /mnt/pve/Homes/bapun/Data/logs/ -type f -name "*.out" -mtime +3 -delete
find /mnt/pve/Homes/bapun/Data/logs/ -type f -name "*.err" -mtime +3 -delete

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

# --- Run: 30 subjects total (15 parallel workers), each with 5-core inner jobs ---
python ../fit_rnn.py \
    --max-subjects 30 \
    --n-jobs-subject 15 \
    --n-jobs-inner 5

# --- Log job end ---
echo "Job finished at: $(date)"