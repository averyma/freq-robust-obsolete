#!/bin/bash
project="wandb_project_name"
gpu="t4v2"
enable_wandb=true #true/false

for lr in 0.01; do
	bash launch_slurm_job.sh ${gpu} jobname_${lr} 1 "python3 main.py --method \"standard\" --lr ${lr} --dataset \"cifar10\" --epoch 10 --wandb_project \"${project}\" --enable_wandb ${enable_wandb}"
done
