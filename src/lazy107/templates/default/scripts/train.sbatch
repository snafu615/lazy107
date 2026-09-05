#!/bin/bash
# Example Slurm script; lazy107 rewrites this file on `render`.
#SBATCH --job-name=train
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate {project}
set -u

python src/train.py
