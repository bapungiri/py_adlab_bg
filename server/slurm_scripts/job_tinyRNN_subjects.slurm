#!/bin/bash
# One-node run: 4 subjects in parallel, each with 12 cores
#SBATCH --job-name=tinyRNN_4x12
#SBATCH --cpus-per-task=48         # total CPUs used by this job
#SBATCH --mem=96G                  # leave ~32G RAM free for others
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# --- Timestamp for logs ---
basefolder=/mnt/pve/Homes/bapun/Data/
timestamp=$(date +"%Y%m%d_%H%M%S")
logsfolder=${basefolder}/logs
mkdir -p ${logsfolder}

#--- logs cleanup -----
find /mnt/pve/Homes/bapun/Data/logs/ -type f -name "*.out" -mtime +7 -delete
find /mnt/pve/Homes/bapun/Data/logs/ -type f -name "*.err" -mtime +7 -delete

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

# --- Run pilot: 4 subjects in parallel, each with 12-core inner jobs ---
python ../tinyRNN_2Arm_subjects.py \
    --max-subjects 12 \
    --n-jobs-subject 4 \
    --n-jobs-inner 12

# --- Log job end ---
echo "Job finished at: $(date)"


