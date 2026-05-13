#!/bin/bash

# Activate the conda environment
source /home/falcon/student1/miniconda3/etc/profile.d/conda.sh
conda activate mldl

# Restrict the script to ONLY use GPU 1
export CUDA_VISIBLE_DEVICES=1

# Run the python script in the background
echo "Starting Experiment 4 (100 Epochs) in the background on GPU 1..."
nohup python src/train.py --config configs/train_config_exp4.yaml > logs/train_output_exp4.log 2>&1 &

echo "Success! The script is running."
echo "You can watch the live terminal output by running:"
echo "tail -f logs/train_output_exp4.log"
