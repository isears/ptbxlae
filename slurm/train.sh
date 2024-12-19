#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=12G
#SBATCH --time=48:00:00
#SBATCH --output ./slurmlogs/train.log

if [ -z "$1" ]
  then
    echo "Supply config as first argument"
    exit 0
fi

# module load cuda/11.3.1
# module load cudnn/8.2.0
nvidia-smi

export PYTHONUNBUFFERED=TRUE


python --version

python main.py --config $1 fit