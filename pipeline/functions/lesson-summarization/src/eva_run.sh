#!/bin/bash

export CLASSPATH=$(pwd)/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar
cd ./src

## Variables
MODE=$1
OUTPUT_FOLDER=$3
CURR_MLFLOW_RUN=$2
SAVED_MLFLOW_RUN="eva_train_$4"
STEPS=1000
PRETRAINED_PRESUMM_PATH="../models/bertextabs.pt"

echo $MODE

if [ $MODE = "train" ]
then
  TRAIN_FOLDER=$OUTPUT_FOLDER
elif [ $MODE = "predict" ]
then
  FORECAST_FOLDER=$OUTPUT_FOLDER
fi



FINETUNED_MODEL_PATH="../models/${TRAIN_FOLDER}.pt"
LOG_NAME="${OUTPUT_FOLDER}.log"
FINETUNED_MODEL="../models/${TRAIN_FOLDER}.pt/model_step_$((148000+STEPS)).pt"
#FINETUNED_MODEL="../models/${TRAIN_FOLDER}.pt/model_step_148100.pt"

## Preprocessing

### Splitting and Tokenization
echo [INFO] Start splitting and tokenization
mkdir -p "../merged_stories_tokenized/$OUTPUT_FOLDER/"
echo [INFO] dummy
echo 'dummy text' > "../logs/$LOG_NAME"
echo [INFO] Start tokenization
python preprocess.py -mode tokenize -raw_path "../raw_data/$OUTPUT_FOLDER/" -save_path "../merged_stories_tokenized/$OUTPUT_FOLDER/" -log_file "../logs/$LOG_NAME"

### Simpler json files
echo [INFO] Start creating simpler json files
mkdir -p "../json_data/$OUTPUT_FOLDER/"
python preprocess.py -mode format_to_lines -raw_path "../merged_stories_tokenized/$OUTPUT_FOLDER/" -save_path "../json_data/$OUTPUT_FOLDER/" -n_cpus 1 -use_bert_basic_tokenizer false -log_file "../logs/$LOG_NAME" -shard_size 20000

### Pytorch files
echo [INFO] Start creating Pytorch files
mkdir -p "../bert_data/$OUTPUT_FOLDER/"
python preprocess.py -mode format_to_bert -raw_path "../json_data/$OUTPUT_FOLDER/" -save_path "../bert_data/$OUTPUT_FOLDER/" -lower -n_cpus 1 -log_file "../logs/$LOG_NAME" -min_src_nsents 1 -max_src_nsents 500 -min_src_ntokens_per_sent 3 -max_src_ntokens_per_sent 500 -shard_size 20000

## Training
if [ $MODE = "train" ]
then
  echo [INFO] Start training
  python train.py -mlflow_run $CURR_MLFLOW_RUN -task abs -mode train -bert_data_path "../bert_data/$TRAIN_FOLDER/" -dec_dropout 0.2  -model_path "$FINETUNED_MODEL_PATH" -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 100 -batch_size 4 -train_steps $((148000+STEPS)) -report_every 20 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 100 -warmup_steps_dec 50 -max_pos 512 -visible_gpus 0 -log_file "../logs/$LOG_NAME"  -train_from ../models/bertextabs.pt
elif [ $MODE = "predict" ]
then
  echo [INFO] Start testing
  ### Prepare data
  cp -r "../bert_data/$FORECAST_FOLDER/.train.0.bert.pt" "../bert_data/$FORECAST_FOLDER/.bert.0.test.pt"
  python train.py -mlflow_run $CURR_MLFLOW_RUN -train_id $4 -task abs -mode test -batch_size 16 -test_batch_size 500 -bert_data_path "../bert_data/$FORECAST_FOLDER/.bert.0" -log_file "../logs/$LOG_NAME" -model_path "$FINETUNED_MODEL_PATH" -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 10 -result_path "../logs/$LOG_NAME" -test_from "$FINETUNED_MODEL"
fi


