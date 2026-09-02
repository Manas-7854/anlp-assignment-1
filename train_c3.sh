#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -p u22
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=anlp_c3
#SBATCH --output=c3_log.txt
#SBATCH --error=c3_log.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=manas.agrawal@research.iiit.ac.in

set -e
cd "$SLURM_SUBMIT_DIR"

echo "[$(date)] Job ${SLURM_JOB_ID:-local}: starting C3 training"
echo "Working directory: $(pwd)"
echo "Activating Conda environment: torch_env"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch_env

echo "[$(date)] Environment ready; launching C3 with batch size 2"
echo "Python interpreter: $CONDA_PREFIX/bin/python3"
"$CONDA_PREFIX/bin/python3" -u src/train.py C3 --batch-size 2
echo "[$(date)] C3 training completed successfully"
