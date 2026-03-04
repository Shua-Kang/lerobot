lerobot-eval \
  --output_dir=/logs/ \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.batch_size=1 \
  --eval.n_episodes=100 \
  --policy.path=/scratch/qhv6ku/delta/vla/lerobot/xh_dev/train_with_lerobot/sbatch_scripts/logs/libero_spatial_20260301_174032/checkpoints/last/pretrained_model \
  --policy.n_action_steps=10 \
  --output_dir=./eval_logs/ \
  --env.max_parallel_tasks=1