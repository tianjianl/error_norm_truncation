#!/bin/bash

# module load gcc/9.3.0
# module load cuda/11.8.0

echo "now pre-training gpt-2 on wikitext-103, using 4 gpus"
SAVE_DIR=/scratch4/danielk/tli104/gpt2/baseline-test-clm-ent-frac
mkdir -p $SAVE_DIR
accelerate launch run_clm_no_trainer.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --config_name gpt2-large \
    --tokenizer_name gpt2-large \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --output_dir $SAVE_DIR \
    --use_token_pruning \
    --prune_fraction $1 \
    --num_train_epochs 5  

