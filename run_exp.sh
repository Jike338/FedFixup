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



# python3 main_fl_clean_step1.py --dataset cifar10 -a resnet20 --print-freq 100 --steps 1  -b 20 --frac 1 --method step --alpha 0 --rounds 64000 --num_data 49000 --lr 0.2 --num_users 5 --norm_layer BatchNorm --probe_activs --momentum 0 --log_every_round 200 --id exp/tensorboard/BN_step_1_shards_lr02 > exp/BN_step_1_shards_lr02.txt
# python3 main_fl_clean.py --dataset cifar10 -a resnet20 --print-freq 100 --steps 20  -b 20 --frac 1 --method step --alpha 0 --rounds 3200 --num_data 49000 --lr 0.02 --num_users 5 --norm_layer BatchNorm --probe_activs --momentum 0 --log_every_round 5 --id exp/tensorboard/BN_step_20_shards > exp/BN_step_20_shards.txt
# python3 main_fl_clean.py --dataset cifar10 -a resnet20 --print-freq 100 --steps 100  -b 20 --frac 1 --method step --alpha 0 --rounds 640 --num_data 49000 --lr 0.02 --num_users 5 --norm_layer BatchNorm --probe_activs --momentum 0 --log_every_round 1 --id exp/tensorboard/BN_step_100_shards > exp/BN_step_100_shards.txt
# python3 main_fl_clean.py --dataset cifar10 -a resnet20 --print-freq 100 --steps 500  -b 20 --frac 1 --method step --alpha 0 --rounds 128 --num_data 49000 --lr 0.02 --num_users 5 --norm_layer BatchNorm --probe_activs --momentum 0 --log_every_round 1 --id exp/tensorboard/BN_step_500_shards > exp/BN_step_500_shards.txt
# python3 main_fl_clean.py --dataset cifar10 -a resnet20 --print-freq 100 --steps 2500  -b 20 --frac 1 --method step --alpha 0 --rounds 128 --num_data 49000 --lr 0.02 --num_users 5 --norm_layer BatchNorm --probe_activs --momentum 0 --log_every_round 1 --id exp/tensorboard/BN_step_2500_shards_3 > exp/BN_step_2500_shards_3.txt

#large bs
# python3 main_fl_clean_step1.py --dataset cifar10 -a resnet20 --print-freq 100 --steps 1  -b 9000 --frac 1 --method step --alpha 0 --rounds 64000 --num_data 49000 --lr 0.2 --num_users 5 --norm_layer BatchNorm --probe_activs --momentum 0 --log_every_round 200 --id exp/tensorboard/BN_step_1_shards_lr02_bs9000 > exp/BN_step_1_shards_lr02_bs9000.txt
# python3 main_fl_clean.py --dataset cifar10 -a resnet20 --print-freq 100 --steps 20  -b 9000 --frac 1 --method step --alpha 0 --rounds 3200 --num_data 49000 --lr 0.02 --num_users 5 --norm_layer BatchNorm --probe_activs --momentum 0 --log_every_round 5 --id exp/tensorboard/BN_step_20_shards_bs9000 > exp/BN_step_20_shards_bs9000.txt
# python3 main_fl_clean.py --dataset cifar10 -a resnet20 --print-freq 100 --steps 100  -b 9000 --frac 1 --method step --alpha 0 --rounds 640 --num_data 49000 --lr 0.02 --num_users 5 --norm_layer BatchNorm --probe_activs --momentum 0 --log_every_round 1 --id exp/tensorboard/BN_step_100_shards_bs9000 > exp/BN_step_100_shards_bs9000.txt
# python3 main_fl_clean.py --dataset cifar10 -a resnet20 --print-freq 100 --steps 500  -b 9000 --frac 1 --method step --alpha 0 --rounds 128 --num_data 49000 --lr 0.02 --num_users 5 --norm_layer BatchNorm --probe_activs --momentum 0 --log_every_round 1 --id exp/tensorboard/BN_step_500_shards_bs9000 > exp/BN_step_500_shards_bs9000.txt
python3 main_fl_clean.py --dataset cifar10 -a resnet20 --print-freq 100 --steps 2500  -b 9000 --frac 1 --method step --alpha 0 --rounds 128 --num_data 49000 --lr 0.02 --num_users 5 --norm_layer BatchNorm --probe_activs --momentum 0 --log_every_round 1 --id exp/tensorboard/BN_step_2500_shards_bs9000 > exp/BN_step_2500_shards_bs9000.txt