#!/bin/bash
# One simulated subject per array task; submit with submit_model_recovery_array.sh.
# Settings come from the environment (set by the submit script):
#   MODELS, N_SUBJECTS, N_STARTS, EXTRA_ARGS (e.g. '--blocks-per-session 5 --tag window5')
#SBATCH --job-name=ModelRecArray
#SBATCH --cpus-per-task=6
#SBATCH --mem=1200M
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

logsfolder=/mnt/pve/Homes/bapun/Data/logs
mkdir -p ${logsfolder}
exec > ${logsfolder}/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.stream.log
exec 2>&1
echo "Job started at: $(date) on $(hostname); array ${SLURM_ARRAY_JOB_ID} task ${SLURM_ARRAY_TASK_ID}"

eval "$(/mnt/pve/Homes/conda/miniconda3/bin/conda shell.bash hook)"
conda activate bapun_conda_env
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH=/mnt/pve/Homes/bapun/Codes/NeuroPy:/mnt/pve/Homes/bapun/Codes/BanditPy:/mnt/pve/Homes/bapun/Codes/py_adlab_bg:$PYTHONPATH

cd /mnt/pve/Homes/bapun/Codes/py_adlab_bg/server
python model_recovery.py --models ${MODELS} --n-subjects "$N_SUBJECTS" --n-starts "$N_STARTS" \
    --n-jobs-inner "${SLURM_CPUS_PER_TASK}" --task-index "${SLURM_ARRAY_TASK_ID}" ${EXTRA_ARGS}

status=$?  # keep python's exit code so SLURM marks failures FAILED
echo "Job finished at: $(date) (exit $status)"
exit $status
