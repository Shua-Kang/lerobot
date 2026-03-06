#!/bin/bash

#Submit this script with: sbatch myjob.slurm

#SBATCH --time=72:00:00   # job time limit
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --cpus-per-task=12   # number of CPU cores per task
#SBATCH --gres=gpu:a100:1   # gpu devices per node
#SBATCH --constraint=a100_80gb
#SBATCH --partition=gpu   # partition
#SBATCH --mail-user=xuhui.kang@outlook.com   # email address
#SBATCH --mail-type=FAIL
#SBATCH --account=lia-lab-members   # allocation name
#SBATCH --output=sbatch_logs/slurm_logs/%x_%j.out
#SBATCH --error=sbatch_logs/slurm_logs/%x_%j.err
#SBATCH --mem=70G

# ============================================================
# Continual training: libero_10 checkpoint -> libero_goal
# ============================================================
# Set this to the checkpoint path from the libero_10 training run.
# It should point to the pretrained_model directory inside a checkpoint.
# Example: ./logs/libero10_20260301_120000/checkpoints/last/pretrained_model
#          ./logs/libero10_20260301_120000/checkpoints/100000/pretrained_model
CHECKPOINT_PATH=/home/hui/lerobot/xh_dev/nebius/scripts/checkpoints/lebero_10_40k_64batchsize/pretrained_model

nvidia-smi

lerobot-train \
  --policy.path=${CHECKPOINT_PATH} \
  --policy.repo_id=${HF_USER}/libero-test-libero-goal-continual \
  --policy.push_to_hub=false \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --num_workers=16 \
  --env.type=libero \
  --env.task=libero_goal \
  --output_dir=./logs/liberogoal_continual_$(date +%Y%m%d_%H%M%S)/ \
  --steps=40000 \
  --batch_size=64 \
  --env.disable_env_checker=true \
  --eval.batch_size=5 \
  --eval.n_episodes=20 \
  --eval_freq=0 \
  --wandb.enable=true \
  --job_name=smolvla_liberogoal_continual_1gpu
