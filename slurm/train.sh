#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output ./slurmlogs/train.log

export PYTHONUNBUFFERED=TRUE

if [ -z "$1" ]
  then
    echo "Supply config as first argument"
    exit 0
fi

nvidia-smi
python --version

# python main.py --config $1 fit --ckpt_path cache/savedmodels/synthetic-oscar.ckpt
python main.py --config $1 fit