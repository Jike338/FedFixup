#!/bin/bash

#SBATCH --account=kpsounis_171
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=a100:2
#SBATCH --output=slurm_out/%x_%j.out


module load python
module spider cuda

cd /scratch1/jikezhon/FedFixup
source ~/miniconda3/bin/activate fl

python3 main_fl_clean.py --dataset cifar10 -a resnet18 --print-freq 100 --steps 20 -b 20 --frac 0.5 --method dir --alpha 0.1 --rounds 3200 --freeze_bn_at_round 1600 --num_classes 10 --num_data 49000 --lr 0.02 --num_users 10 --norm_layer BatchNorm --probe_activs --id ablation_main/tensorboard/resnet18_bn_cifar10_delay_freq20_a01 > ablation_main/resnet18_bn_cifar10_delay_freq20_a01.txt
python3 main_fl_clean.py --dataset cifar10 -a resnet18 --print-freq 100 --steps 500 -b 20 --frac 0.5 --method dir --alpha 0.1 --rounds 128 --freeze_bn_at_round 64 --num_classes 10 --num_data 49000 --lr 0.02 --num_users 10 --norm_layer BatchNorm --probe_activs --id ablation_main/tensorboard/resnet18_bn_cifar10_delay_freq500_a01 > ablation_main/resnet18_bn_cifar10_delay_freq500_a01.txt
python3 main_fl_clean.py --dataset cifar10 -a resnet18 --print-freq 100 --steps 2500 -b 20 --frac 0.5 --method dir --alpha 0.1 --rounds 128 --freeze_bn_at_round 13 --num_classes 10 --num_data 49000 --lr 0.02 --num_users 10 --norm_layer BatchNorm --probe_activs --id ablation_main/tensorboard/resnet18_bn_cifar10_delay_freq2500_a01 > ablation_main/resnet18_bn_cifar10_delay_freq2500_a01.txt