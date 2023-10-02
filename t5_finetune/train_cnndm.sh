#!/bin/bash

echo "now training cnndm on gpu $1"
rm -rf /scratch/tli104/cnndm
mkdir -p /scratch/tli104/cnndm 

CUDA_VISIBLE_DEVICES=$1 python run_summarization_no_trainer.py \
    --model_name_or_path t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /scratch/tli104/cnndm --seed 12 --num_beams 5 
