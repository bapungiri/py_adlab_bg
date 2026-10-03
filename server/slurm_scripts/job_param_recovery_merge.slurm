#!/bin/bash
# Merge a recovery array run's partials into one GroupData save.
# usage (by submit_recovery_array.sh): DESIGN=.. VARIANT=.. MAX_SUBJECTS=.. \
#   sbatch --dependency=afterok:<array id> job_param_recovery_merge.slurm <array id>
#SBATCH --job-name=RecMerge
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
python param_recovery.py --policy "${POLICY:-Qlearn}" --design "$DESIGN" --variant "$VARIANT" \
    --max-subjects "$MAX_SUBJECTS" --merge "$1" ${EXTRA_ARGS}
echo "Merge finished at: $(date)"
