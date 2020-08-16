#!/bin/bash

export CLASSPATH=$(pwd)/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar
cd ./src

## Variables
TRAIN_FOLDER=$1
CURR_MLFLOW_RUN=$2
STEPS=1000
PRETRAINED_PRESUMM_PATH="../models/bertextabs.pt"
LOG_NAME="${TRAIN_FOLDER}.log"
FINETUNED_MODEL_PATH="../models/${TRAIN_FOLDER}.pt"
#FINETUNED_MODEL="../models/${TRAIN_FOLDER}.pt/model_step_148100.pt"

## Preprocessing

### Splitting and Tokenization
echo [INFO] Start splitting and tokenization
mkdir -p "../merged_stories_tokenized/$TRAIN_FOLDER/"
echo [INFO] dummy
echo 'dummy text' > "../logs/$LOG_NAME"
echo [INFO] Start tokenization
python preprocess.py -mode tokenize -raw_path "../raw_data/$TRAIN_FOLDER/" -save_path "../merged_stories_tokenized/$TRAIN_FOLDER/" -log_file "../logs/$LOG_NAME"

### Simpler json files
echo [INFO] Start creating simpler json files
mkdir -p "../json_data/$TRAIN_FOLDER/"
python preprocess.py -mode format_to_lines -raw_path "../merged_stories_tokenized/$TRAIN_FOLDER/" -save_path "../json_data/$TRAIN_FOLDER/" -n_cpus 1 -use_bert_basic_tokenizer false -log_file "../logs/$LOG_NAME" -shard_size 20000

### Pytorch files
echo [INFO] Start creating Pytorch files
mkdir -p "../bert_data/$TRAIN_FOLDER/"
python preprocess.py -mode format_to_bert -raw_path "../json_data/$TRAIN_FOLDER/" -save_path "../bert_data/$TRAIN_FOLDER/" -lower -n_cpus 1 -log_file "../logs/$LOG_NAME" -min_src_nsents 1 -max_src_nsents 500 -min_src_ntokens_per_sent 3 -max_src_ntokens_per_sent 500 -shard_size 20000

## Training
echo [INFO] Start training
python train.py -mlflow_run $CURR_MLFLOW_RUN -task abs -mode train -bert_data_path "../bert_data/$TRAIN_FOLDER/" -dec_dropout 0.2  -model_path "$FINETUNED_MODEL_PATH" -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 100 -batch_size 4 -train_steps $((148000+STEPS)) -report_every 20 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 100 -warmup_steps_dec 50 -max_pos 512 -visible_gpus 0 -log_file "../logs/$LOG_NAME"  -train_from ../models/bertextabs.pt


