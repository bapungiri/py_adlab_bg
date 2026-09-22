#!/bin/bash
#SBATCH --job-name=rnn_models_job
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00                    # Max runtime: 3 days
#SBATCH --nodelist=server1,server2,server3   # Run on a specific node (optional)

# --- Timestamp for logs ---
basefolder=/mnt/pve/Homes/bapun/Data/rnn_data
timestamp=$(date +"%Y%m%d_%H%M%S")
logsfolder=${basefolder}/logs
mkdir -p ${logsfolder}

# --- Redirect stdout and stderr ---
exec > ${logsfolder}/${SLURM_JOB_NAME}_${timestamp}_${SLURM_JOB_ID}.out
exec 2> ${logsfolder}/${SLURM_JOB_NAME}_${timestamp}_${SLURM_JOB_ID}.err

# --- Log job start ---
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "SLURM Job ID: $SLURM_JOB_ID"

# --- Conda activation ---
eval "$(/mnt/pve/Homes/conda/miniconda3/bin/conda shell.bash hook)"
conda activate bapun_conda_env

# --- Export custom PYTHONPATH ---
export PYTHONPATH=/mnt/pve/Homes/bapun/Codes/NeuroPy:/mnt/pve/Homes/bapun/Codes/BanditPy:/mnt/pve/Homes/bapun/Codes/py_adlab_bg:$PYTHONPATH

# --- Run your Python job ---
python create_models.py

# --- Log job end ---
echo "Job finished at: $(date)"
