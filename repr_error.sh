#!/bin/bash

rollout_paths=("Ant-v4" "HalfCheetah-v4" "InvertedPendulum-v4" "Humanoid-v4")
tasks=("policy" "value" "reward" "model")
hiddens=("48" "32" "64")
sizes=("3e4" "1e5" "3e5")
layers=("2" "3")
num_runs=4
cnt=0

# Run experiments for 4 seeds for each task on 4 GPUs.
for size in "${sizes[@]}"; do
    for hidden in "${hiddens[@]}"; do
        for path in "${rollout_paths[@]}"; do
            for layer in "${layers[@]}"; do
                for task in "${tasks[@]}"; do
                    for ((i = 0; i < num_runs; i++)); do
                        python repr_error.py --env "$path" --type "$task" --seed "$i" --hidden "$hidden" --layer "$layer" --device "0" --size "$size" 
                        cnt=$((cnt+1))
                        echo "Finished $cnt runs"
                    done
                done
            done
        done
    done
done