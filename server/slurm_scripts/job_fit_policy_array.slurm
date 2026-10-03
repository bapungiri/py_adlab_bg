#!/bin/bash
# One session (animal) per array task; submit with submit_fit_array.sh <preset>.
# Each task writes a partial result; job_fit_policy_merge.slurm combines them.
#SBATCH --job-name=FitArray
#SBATCH --cpus-per-task=6         # even (whole cores); n_starts a multiple of this
#SBATCH --mem=4G
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

logsfolder=/mnt/pve/Homes/bapun/Data/logs
mkdir -p ${logsfolder}
exec > ${logsfolder}/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.stream.log
exec 2>&1

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "SLURM array job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID} (job ${SLURM_JOB_ID})"

eval "$(/mnt/pve/Homes/conda/miniconda3/bin/conda shell.bash hook)"
conda activate bapun_conda_env

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=/mnt/pve/Homes/bapun/Codes/NeuroPy:/mnt/pve/Homes/bapun/Codes/BanditPy:/mnt/pve/Homes/bapun/Codes/py_adlab_bg:$PYTHONPATH

cd /mnt/pve/Homes/bapun/Codes/py_adlab_bg/server
python fit_policy_wrapper.py --preset "$1" --task-index "${SLURM_ARRAY_TASK_ID}"

echo "Job finished at: $(date)"
