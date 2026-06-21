#!/bin/bash
# 只用来创建数据集，所以加上debug=default，不然写入日志的时候不太方便解析
# Function to run experiments for a dataset
run_dataset_experiments() {
    local dataset=$1
    local gpu_device=$2
    local bias_values=("${@:3}")
    
    echo "Starting experiments for $dataset on GPU $gpu_device"
    
    for bias in "${bias_values[@]}"; do
        echo "${dataset}_bias_strength_${bias}"
        CUDA_VISIBLE_DEVICES=$gpu_device python src/train.py experiment=${dataset}_grsna data.bias_strength=$bias debug=default
    done
    
    echo "Completed experiments for $dataset"
}

# Define bias strength values for each dataset
aids_bias_values=(0 0.01 0.1 10 9999)
imdb_bias_values=(0 0.01 0.1 10 9999)
p53_bias_values=(0 0.01 0.1 10 9999)
proteins_bias_values=(0 0.01 0.1 10 100 9999)

echo "Starting parallel sensitivity experiments..."

# Run each dataset in parallel as background processes
run_dataset_experiments "aids" 1 "${aids_bias_values[@]}" &
aids_pid=$!

run_dataset_experiments "imdb" 1 "${imdb_bias_values[@]}" &
imdb_pid=$!

run_dataset_experiments "p53" 1 "${p53_bias_values[@]}" &
p53_pid=$!

run_dataset_experiments "proteins" 4 "${proteins_bias_values[@]}" &
proteins_pid=$!

# Wait for all background processes to complete
echo "Waiting for all experiments to complete..."
wait $aids_pid
echo "AIDS experiments completed"

wait $imdb_pid
echo "IMDB experiments completed"

wait $p53_pid
echo "P53 experiments completed"

wait $proteins_pid
echo "Proteins experiments completed"

echo "All sensitivity experiments completed!"