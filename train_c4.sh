#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -p u22
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --job-name=anlp_c4
#SBATCH --output=anlp_c4_%j.log
#SBATCH --error=anlp_c4_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=manas.agrawal@research.iiit.ac.in

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch_env

python src/train.py C4
