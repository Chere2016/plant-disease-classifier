#!/bin/bash

# Activate the conda environment
source /home/falcon/student1/miniconda3/etc/profile.d/conda.sh
conda activate mldl

# Restrict the script to ONLY use GPU 1 (the second A100 GPU)
export CUDA_VISIBLE_DEVICES=1

# Run the python script in the background using nohup. 
# This prevents it from dying if your SSH/terminal session disconnects.
echo "Starting training in the background on GPU 1..."
nohup python3 src/train.py > train_output.log 2>&1 &

echo "Success! The script is running."
echo "You can watch the live terminal output by running:"
echo "tail -f train_output.log"
