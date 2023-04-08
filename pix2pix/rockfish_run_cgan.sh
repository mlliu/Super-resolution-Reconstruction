#!/bin/bash

#SBATCH -A yqiao_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name="cgan"
#SBATCH --mem-per-cpu=100G

module load anaconda

# init virtual environment if needed
conda create -n cgan python=3.7

conda activate cgab # open the Python environment

pip install -r requirements.txt # install Python dependencies

# runs your code
#srun python classification.py  --experiment "overfit"  --device cuda --model "distilbert-base-uncased" --batch_size "64" --lr 1e-4 --num_epochs 30

norm_type="max"
srun python train.py --cuda --norm_type $norm_type --nepochs 2 #--mip_type $mip_type
srun python test.py --cuda --norm_type $norm_type --nepochs 2 #--mip_type $mip_type