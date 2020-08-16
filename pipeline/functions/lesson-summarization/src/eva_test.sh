#!/bin/bash

## Variables
TEST_FOLDER=$1
STEPS=1000
PRETRAINED_PRESUMM_PATH="../models/bertextabs.pt"
LOG_NAME="${TEST_FOLDER}.log"
FINETUNED_MODEL_PATH="../models/${TEST_FOLDER}.pt"
# FINETUNED_MODEL="../models/${TEST_FOLDER}.pt/model_step_148100.pt"
FINETUNED_MODEL="../models/${TEST_FOLDER}.pt/model_step_$((148000+STEPS)).pt"

## Inference

### Prepare data
cp "../bert_data/$TEST_FOLDER/.train.0.bert.pt" "../bert_data/$TEST_FOLDER/.bert.0.test.pt"

## Generate titles
python train.py -task abs -mode test -batch_size 16 -test_batch_size 500 -bert_data_path "../bert_data/$TEST_FOLDER/.bert.0" -log_file "../logs/$LOG_NAME" -model_path "$FINETUNED_MODEL_PATH" -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 10 -result_path "../logs/$LOG_NAME" -test_from "$FINETUNED_MODEL"
