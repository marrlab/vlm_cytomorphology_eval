#!/bin/bash

#SBATCH -o logs/finetuneDino_out.txt
#SBATCH -e logs/finetuneDino_err.txt
#SBATCH -p gpu_p
#SBATCH --time=01:00:00
#SBATCH --nice=10000
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -q gpu_normal


if [ -z "$N" ]; then
  echo "Error: N is not set. Use sbatch --export=N=VALUE run_finetuneDino.sh"
  exit 1
fi

# Environment setup
source $HOME/.bashrc

# choose the right directory
cd /lustre/groups/labs/marr/qscd01/workspace/furkan.dasdelen/vlm_cytomorphology_eval
export PYTHONPATH=/lustre/groups/labs/marr/qscd01/workspace/furkan.dasdelen/vlm_cytomorphology_eval:$PYTHONPATH

# activate your environment
conda activate superbloom

# set checkpoint to evaluate as input 
python -u fine_tune_Dino.py --train_csv /lustre/groups/labs/marr/qscd01/projects/cytology_vlm_eval/Datasets/Acevedo/train/fine_tuning_Acevedo_train_n_per_label_${N}.csv \
							--val_csv /lustre/groups/labs/marr/qscd01/projects/cytology_vlm_eval/Datasets/Acevedo/val/Acevedo_val_labels.csv \
							--test_csv /lustre/groups/labs/marr/qscd01/projects/cytology_vlm_eval/Datasets/Acevedo/test/Acevedo_test_labels.csv \
							--epochs 100 \
							--modelname dinov2_vits14 \
							--results_path Results \
							> logs/output_finetuneDino.txt
