module load miniforge/24.3.0-py3.11 
conda activate /sfs/weka/scratch/qhv6ku/.conda/envs/lerobot
lerobot-train \
  --policy.type=smolvla \
  --policy.repo_id=${HF_USER}/libero-test \
  --policy.load_vlm_weights=true \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --env.type=libero \
  --env.task=libero_10 \
  --output_dir=./logs/temp223/ \
  --steps=10000 \
  --batch_size=64 \
  --eval.batch_size=10 \
  --eval.n_episodes=20 \
  --eval_freq=1000 \
  --wandb.enable=true \
  --job_name=act_so101_test