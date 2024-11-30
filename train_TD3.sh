#!/bin/bash

rollout_paths=("Ant-v4" "HalfCheetah-v4" "InvertedPendulum-v4" "Humanoid-v4")

for env_path in "${rollout_paths[@]}"; do
	python train_TD3.py --policy "TD3" --seed 2 --save_model --env "$env_path"
done
