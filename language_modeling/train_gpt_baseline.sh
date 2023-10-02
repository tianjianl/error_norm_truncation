#!/bin/bash

# module load gcc/9.3.0
# module load cuda/11.8.0

echo "now pre-training gpt2-large on wikitext-103"
SAVE_DIR=/scratch4/danielk/tli104/gpt2/baseline-test-clm-temp
mkdir -p $SAVE_DIR
accelerate launch run_clm_no_trainer.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --config_name gpt2-large \
    --tokenizer_name gpt2-large \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --output_dir $SAVE_DIR \
    --num_train_epochs 5 \

