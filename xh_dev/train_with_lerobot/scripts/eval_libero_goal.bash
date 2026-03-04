lerobot-eval \
  --output_dir=/logs/ \
  --env.type=libero \
  --env.task=libero_goal \
  --eval.batch_size=1 \
  --eval.n_episodes=100 \
  --policy.path=/scratch/qhv6ku/delta/vla/lerobot/xh_dev/train_with_lerobot/sbatch_scripts/logs/libero_goal_20260301_174034/checkpoints/last/pretrained_model \
  --policy.n_action_steps=10 \
  --output_dir=./eval_logs/ \
  --env.max_parallel_tasks=1