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

# python3 main_fl_clean_step1_local_mom.py --dataset cifar10 -a mobilenetv1 --print-freq 100 --steps 1  -b 20 --frac 1 --method step --alpha 0 --rounds 64000 --num_data 50000 --lr 0.2 --num_users 5 --norm_layer BatchNorm --log_every_round 200 --id rebuttal_mobilenet/tensorboard/BN_noactivs_step_1_shards_mom_lr0.002 > rebuttal_mobilenet/BN_noactivs_step_1_shards_mom_lr0.002.txt
# python3 main_fl_clean.py --dataset cifar10 -a mobilenetv1 --print-freq 100 --steps 20  -b 20 --frac 1 --method step --alpha 0 --rounds 3200 --num_data 50000 --lr 0.02 --num_users 5 --norm_layer BatchNorm --log_every_round 20 --id rebuttal_mobilenet/tensorboard/BN_noactivs_step_20_shards_lr0.02 > rebuttal_mobilenet/BN_noactivs_step_20_shards_lr0.02.txt
# python3 main_fl_clean.py --dataset cifar10 -a mobilenetv1 --print-freq 100 --steps 100  -b 20 --frac 1 --method step --alpha 0 --rounds 640 --num_data 50000 --lr 0.02 --num_users 5 --norm_layer BatchNorm --log_every_round 5 --id rebuttal_mobilenet/tensorboard/BN_noactivs_step_100_shards_lr0.02 > rebuttal_mobilenet/BN_noactivs_step_100_shards_lr0.02.txt
# python3 main_fl_clean.py --dataset cifar10 -a mobilenetv1 --print-freq 100 --steps 500  -b 20 --frac 1 --method step --alpha 0 --rounds 128 --num_data 50000 --lr 0.002 --num_users 5 --norm_layer BatchNorm --log_every_round 1 --id rebuttal_mobilenet/tensorboard/BN_noactivs_step_500_shards_lr0.002 > rebuttal_mobilenet/BN_noactivs_step_500_shards_lr0.002.txt
# python3 main_fl_clean.py --dataset cifar10 -a mobilenetv1 --print-freq 100 --steps 2500  -b 20 --frac 1 --method step --alpha 0 --rounds 128 --num_data 50000 --lr 0.002 --num_users 5 --norm_layer BatchNorm --log_every_round 1 --id rebuttal_mobilenet/tensorboard/BN_noactivs_step_2500_shards_lr0.002 > rebuttal_mobilenet/BN_noactivs_step_2500_shards_lr0.002.txt

# python3 main_fl_clean_step1_local_mom.py --dataset cifar10 -a mobilenetv1 --print-freq 100 --steps 1  -b 20 --frac 1 --method step --alpha 0 --rounds 64000 --num_data 50000 --lr 0.2 --num_users 5 --norm_layer GroupNorm --log_every_round 200 --id rebuttal_mobilenet/tensorboard/GN_noactivs_step_1_shards_mom_lr0.002 > rebuttal_mobilenet/GN_noactivs_step_1_shards_mom_lr0.002.txt
# python3 main_fl_clean.py --dataset cifar10 -a mobilenetv1 --print-freq 100 --steps 20  -b 20 --frac 1 --method step --alpha 0 --rounds 3200 --num_data 50000 --lr 0.007 --num_users 5 --norm_layer GroupNorm --log_every_round 20 --id rebuttal_mobilenet/tensorboard/GN_noactivs_step_20_shards_lr0.007 > rebuttal_mobilenet/GN_noactivs_step_20_shards_lr0.007.txt
# python3 main_fl_clean.py --dataset cifar10 -a mobilenetv1 --print-freq 100 --steps 100  -b 20 --frac 1 --method step --alpha 0 --rounds 640 --num_data 50000 --lr 0.007 --num_users 5 --norm_layer GroupNorm --log_every_round 5 --id rebuttal_mobilenet/tensorboard/GN_noactivs_step_100_shards2_lr0.007 > rebuttal_mobilenet/GN_noactivs_step_100_shards2_lr0.007.txt
# python3 main_fl_clean.py --dataset cifar10 -a mobilenetv1 --print-freq 100 --steps 500  -b 20 --frac 1 --method step --alpha 0 --rounds 128 --num_data 50000 --lr 0.007 --num_users 5 --norm_layer GroupNorm --log_every_round 1 --id rebuttal_mobilenet/tensorboard/GN_noactivs_step_500_shards2_lr0.007 > rebuttal_mobilenet/GN_noactivs_step_500_shards2_lr0.007.txt
# python3 main_fl_clean.py --dataset cifar10 -a mobilenetv1 --print-freq 100 --steps 2500  -b 20 --frac 1 --method step --alpha 0 --rounds 128 --num_data 50000 --lr 0.002 --num_users 5 --norm_layer GroupNorm --log_every_round 1 --id rebuttal_mobilenet/tensorboard/GN_noactivs_step_2500_shards_lr0.002 > rebuttal_mobilenet/GN_noactivs_step_2500_shards_lr0.002.txt

# python3 main_fl_clean_step1_local_mom.py --dataset cifar10 -a mobilenetv1 --print-freq 100 --steps 1  -b 20 --frac 1 --method step --alpha 0 --rounds 64000 --num_data 50000 --lr 0.2 --num_users 5 --freeze_bn_at_round 32000 --norm_layer BatchNorm --log_every_round 200 --id rebuttal_mobilenet/tensorboard/DelayBN_noactivs_step_1_shards_delay_mom_lr0.002 > rebuttal_mobilenet/DelayBN_noactivs_step_1_shards_delay_mom_lr0.002.txt
# python3 main_fl_clean.py --dataset cifar10 -a mobilenetv1 --print-freq 100 --steps 20  -b 20 --frac 1 --method step --alpha 0 --rounds 3200 --num_data 50000 --lr 0.002 --num_users 5 --freeze_bn_at_round 1600 --norm_layer BatchNorm --log_every_round 20 --id rebuttal_mobilenet/tensorboard/DelayBN_noactivs_step_20_shards_delay_lr0.002 > rebuttal_mobilenet/DelayBN_noactivs_step_20_shards_delay_lr0.002.txt
# python3 main_fl_clean.py --dataset cifar10 -a mobilenetv1 --print-freq 100 --steps 100  -b 20 --frac 1 --method step --alpha 0 --rounds 640 --num_data 50000 --lr 0.002 --num_users 5 --freeze_bn_at_round 320 --norm_layer BatchNorm --log_every_round 5 --id rebuttal_mobilenet/tensorboard/DelayBN_noactivs_step_100_shards_delay_lr0.002 > rebuttal_mobilenet/DelayBN_noactivs_step_100_shards_delay_lr0.002.txt
# python3 main_fl_clean.py --dataset cifar10 -a mobilenetv1 --print-freq 100 --steps 500  -b 20 --frac 1 --method step --alpha 0 --rounds 128 --num_data 50000 --lr 0.002 --num_users 5 --freeze_bn_at_round 64 --norm_layer BatchNorm --log_every_round 1 --id rebuttal_mobilenet/tensorboard/DelayBN_noactivs_step_500_shards_delay_lr0.002 > rebuttal_mobilenet/DelayBN_noactivs_step_500_shards_delay_lr0.002.txt
# python3 main_fl_clean.py --dataset cifar10 -a mobilenetv1 --print-freq 100 --steps 2500  -b 20 --frac 1 --method step --alpha 0 --rounds 128 --num_data 50000 --lr 0.002 --num_users 5 --freeze_bn_at_round 13 --norm_layer BatchNorm --log_every_round 1 --id rebuttal_mobilenet/tensorboard/DelayBN_noactivs_step_2500_shards_3_delay_lr0.002 > rebuttal_mobilenet/DelayBN_noactivs_step_2500_shards_3_delay_lr0.002.txt