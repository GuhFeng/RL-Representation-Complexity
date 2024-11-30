#!/bin/bash

envs=("Ant-v4" "HalfCheetah-v4" "InvertedPendulum-v4" "Humanoid-v4")
sizes=("1e6")

for env in "${envs[@]}"; do
    for size in "${sizes[@]}"; do
        python rollout.py --env "$env" --size "$size" &
    done
done
