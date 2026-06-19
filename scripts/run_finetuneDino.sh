#!/bin/bash
#
# SLURM launcher for DINO fine-tuning on the Acevedo dataset.
#
# Usage:
#   sbatch --export=N=<samples_per_label>,PROJECT_ROOT=<repo_path>,DATA_ROOT=<data_path>,CONDA_ENV=<env_name> \
#       scripts/run_finetuneDino.sh
#
# All paths below are configurable through environment variables so the script
# is portable across machines / clusters.

#SBATCH -o logs/finetuneDino_out.txt
#SBATCH -e logs/finetuneDino_err.txt
#SBATCH -p gpu_p
#SBATCH --time=01:00:00
#SBATCH --nice=10000
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -q gpu_normal

set -euo pipefail

# ----- required -----
if [ -z "${N:-}" ]; then
  echo "Error: N is not set. Use: sbatch --export=N=VALUE scripts/run_finetuneDino.sh"
  exit 1
fi

# ----- configurable paths (override via --export=...) -----
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/Datasets}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/Results}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
DINO_WEIGHTS="${DINO_WEIGHTS:-${PROJECT_ROOT}/DinoBloom-S.pth}"
CONDA_ENV="${CONDA_ENV:-vlm_eval}"

mkdir -p "${LOG_DIR}"

# ----- environment setup -----
source "${HOME}/.bashrc"
conda activate "${CONDA_ENV}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# ----- run -----
# Remove --modelpath if you want to test the Dinov2 baseline.
python -u -m vlm_cytomorphology_eval.finetuning.fine_tune_Dino \
    --train_csv "${DATA_ROOT}/Acevedo/train/fine_tuning_Acevedo_train_n_per_label_${N}.csv" \
    --val_csv   "${DATA_ROOT}/Acevedo/val/Acevedo_val_labels.csv" \
    --test_csv  "${DATA_ROOT}/Acevedo/test/Acevedo_test_labels.csv" \
    --epochs 100 \
    --modelname dinov2_vits14 \
    --modelpath "${DINO_WEIGHTS}" \
    --results_path "${RESULTS_DIR}" \
    > "${LOG_DIR}/output_finetuneDino.txt"
