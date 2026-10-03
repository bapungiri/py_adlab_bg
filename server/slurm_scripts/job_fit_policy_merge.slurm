#!/bin/bash
# Merge an array run's partial results into one GroupData save.
# usage: sbatch --dependency=afterok:<array job id> job_fit_policy_merge.slurm <preset> <array job id>
#SBATCH --job-name=FitMerge
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

logsfolder=/mnt/pve/Homes/bapun/Data/logs
mkdir -p ${logsfolder}
exec > ${logsfolder}/${SLURM_JOB_NAME}_$2_${SLURM_JOB_ID}.stream.log
exec 2>&1
echo "Merge started at: $(date)"

eval "$(/mnt/pve/Homes/conda/miniconda3/bin/conda shell.bash hook)"
conda activate bapun_conda_env
export PYTHONUNBUFFERED=1
export PYTHONPATH=/mnt/pve/Homes/bapun/Codes/NeuroPy:/mnt/pve/Homes/bapun/Codes/BanditPy:/mnt/pve/Homes/bapun/Codes/py_adlab_bg:$PYTHONPATH

cd /mnt/pve/Homes/bapun/Codes/py_adlab_bg/server
python fit_policy_wrapper.py --preset "$1" --merge "$2"
status=$?  # keep python's exit code so SLURM marks failures FAILED
echo "Merge finished at: $(date) (exit $status)"
exit $status
