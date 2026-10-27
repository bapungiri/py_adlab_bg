#!/bin/bash
# Merge a model-recovery array run's partials into one GroupData save.
# usage (by submit_model_recovery_array.sh): MODELS=.. N_SUBJECTS=.. \
#   sbatch --dependency=afterok:<array id> job_model_recovery_merge.slurm <array id>
#SBATCH --job-name=ModelRecMerge
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

logsfolder=/mnt/pve/Homes/bapun/Data/logs
mkdir -p ${logsfolder}
exec > ${logsfolder}/${SLURM_JOB_NAME}_$1_${SLURM_JOB_ID}.stream.log
exec 2>&1
echo "Merge started at: $(date)"

eval "$(/mnt/pve/Homes/conda/miniconda3/bin/conda shell.bash hook)"
conda activate bapun_conda_env
export PYTHONUNBUFFERED=1
export PYTHONPATH=/mnt/pve/Homes/bapun/Codes/NeuroPy:/mnt/pve/Homes/bapun/Codes/BanditPy:/mnt/pve/Homes/bapun/Codes/py_adlab_bg:$PYTHONPATH

cd /mnt/pve/Homes/bapun/Codes/py_adlab_bg/server
python model_recovery.py --models ${MODELS} --n-subjects "$N_SUBJECTS" --merge "$1" ${EXTRA_ARGS}
status=$?  # keep python's exit code so SLURM marks failures FAILED
echo "Merge finished at: $(date) (exit $status)"
exit $status
