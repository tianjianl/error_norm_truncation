#!/bin/bash

echo "now training cnndm on gpu $1, prune fraction $2"
SAVE_DIR=/scratch/tli104/cnndm_token_prune_fraction_$2
rm -rf $SAVE_DIR
mkdir -p $SAVE_DIR
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$1 python run_summarization_no_trainer.py \
    --model_name_or_path t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir $SAVE_DIR \
    --num_beams 5 --seed 12 \
    --token_pruning \
    --prune_fraction $2    
