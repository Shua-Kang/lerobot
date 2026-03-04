#!/bin/bash

#Submit this script with: sbatch myjob.slurm

#SBATCH --time=72:00:00   # job time limit
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --cpus-per-task=6   # number of CPU cores per task
#SBATCH --gres=gpu:a100:1   # gpu devices per node
#SBATCH --partition=gpu   # partition
#SBATCH --constraint=a100_80gb
#SBATCH --mail-user=xuhui.kang@outlook.com   # email address
#SBATCH --mail-type=FAIL
#SBATCH --account=lia-lab-members   # allocation name
#SBATCH --mem=70G
#SBATCH --output=sbatch_logs/slurm_logs/%x_%j.out
#SBATCH --error=sbatch_logs/slurm_logs/%x_%j.err

module load miniforge/24.3.0-py3.11
conda activate /sfs/weka/scratch/qhv6ku/.conda/envs/lerobot

nvidia-smi

lerobot-train \
  --policy.type=smolvla \
  --policy.repo_id=${HF_USER}/libero-test-libero-spatial \
  --policy.load_vlm_weights=true \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --env.type=libero \
  --env.task=libero_spatial \
  --output_dir=./logs/libero_spatial_$(date +%Y%m%d_%H%M%S)/ \
  --steps=100000 \
  --batch_size=32 \
  --eval_freq=0 \
  --wandb.enable=true \
  --job_name=smolvla_libero_spatial_1gpu
