#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -p u22
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=anlp_c5
#SBATCH --output=c5_log.txt
#SBATCH --error=c5_log.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=manas.agrawal@research.iiit.ac.in

set -e
cd "$SLURM_SUBMIT_DIR"

echo "[$(date)] Job ${SLURM_JOB_ID:-local}: starting C5 training"
echo "Working directory: $(pwd)"
echo "Activating Conda environment: torch_env"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch_env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date)] Environment ready; launching C5 final train/validation/test run"
echo "Python interpreter: $CONDA_PREFIX/bin/python3"
"$CONDA_PREFIX/bin/python3" -u src/train.py C5 \
    --device cuda --epochs 20 --batch-size 2 --fp16 --wandb-mode online \
    --learning-rate 3e-4 --warmup-ratio 0.05 --min-learning-rate 1e-5 \
    --max-seq-length 4096 --max-decode-length 4096 --evaluate-test
echo "[$(date)] C5 training completed successfully"
