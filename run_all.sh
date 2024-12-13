#!/bin/bash
#PBS -l walltime=80:15:00
#PBS -l nodes=1:ppn=2:gpus=1,mem=100GB
#PBS -A PAS2099
#PBS -N jike338
#PBS -j oe
#PBS -m be
#SBATCH --output=script_output1/R-%x.%j.out
module load python
module load cuda


cd /fs/ess/scratch/PAS2099/jike/DelayBN



python3 main_fl_clean.py --dataset cifar10 -a resnet20 --print-freq 100 --steps 100  -b 20 --frac 1 --method step --alpha 0 --rounds 640 --num_data 50000 --lr 0.02 --num_users 5 --norm_layer BatchNorm --log_every_round 5 --id test/tensorboard/BN_step_100_shards > test/BN_step_100_shards2.txt
