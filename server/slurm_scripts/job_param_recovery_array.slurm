#!/bin/bash
# One simulated subject per array task; submit with submit_recovery_array.sh.
# Settings come from the environment (set by the submit script):
#   DESIGN, VARIANT, OPTIMIZER, N_STARTS, MAX_SUBJECTS, POLICY (default Qlearn),
#   EXTRA_ARGS (e.g. '--true-range sticky=0,5 --tag truesticky0to5')
#SBATCH --job-name=RecArray
#SBATCH --cpus-per-task=10
#SBATCH --mem=2G
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
python param_recovery.py --policy "${POLICY:-Qlearn}" --design "$DESIGN" --variant "$VARIANT" \
    --optimizer "$OPTIMIZER" --n-starts "$N_STARTS" --max-subjects "$MAX_SUBJECTS" \
    --n-jobs-inner "${SLURM_CPUS_PER_TASK}" --task-index "${SLURM_ARRAY_TASK_ID}" ${EXTRA_ARGS}

echo "Job finished at: $(date)"
