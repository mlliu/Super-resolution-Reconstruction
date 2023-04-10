#!/bin/bash

#SBATCH -A yqiao4
#SBATCH --partition defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name="cgan test"
#SBATCH --mem-per-cpu=10G

module load anaconda

# init virtual environment if needed
#conda create -n cgan python=3.7

conda activate cgan # open the Python environment

#pip install -r requirements.txt # install Python dependencies

# runs your code
#srun python classification.py  --experiment "overfit"  --device cuda --model "distilbert-base-uncased" --batch_size "64" --lr 1e-4 --num_epochs 30
norm_type="max"
mip_type=0
nepochs=20
scratchpath="/home/mliu121/scratch4-yqiao4/"
modelfile=$scratchpath"checkpoint_norm_"$norm_type"_mip_"$mip_type"/"
mkdir -p $modelfile

#srun python train.py --cuda --norm_type $norm_type  --mip_type $mip_type --modelfile $modelfile > $modelfile"train_log" 
srun python test.py --cuda --norm_type $norm_type  --nepochs $nepochs --mip_type $mip_type --modelfile $modelfile > $modelfule"test_log"
srun python mip.py --modelfile $modelfile --nepochs 20