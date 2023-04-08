#!/bin/sh
#$ -l gpu
#$ -l mem_free=100G,h_vmem=150G
#$ -cwd 
#$ -o log
#$ -e log
#$ -m e
#$ -M mliu121@jhu.edu

module load conda/3.0
source activate cgan
module load python/3.9.10
module load cudnn
export CUDA_VISIBLE_DEVICES=0
norm_type="max"
python3 train.py --cuda --norm_type $norm_type #--mip_type $mip_type
python3 test.py --cuda --norm_type $norm_type --nepochs 20 #--mip_type $mip_type