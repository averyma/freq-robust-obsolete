#!/bin/bash
project="wandb_project_name"
gpu="t4v2"
enable_wandb=true #true/false

for method in "standard" "adv"; do
	for lr in 0.01; do
		bash launch_slurm_job.sh ${gpu} job_${method} 1 "python3 main.py --method \"${method}\" --lr ${lr} --dataset \"cifar10\" --epoch 100 --wandb_project \"${project}\" --enable_wandb ${enable_wandb}"
	done
done
