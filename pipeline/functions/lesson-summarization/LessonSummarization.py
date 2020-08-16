import pandas as pd
import pickle
import seaborn as sns
import mlflow
from mlflow import *
# from mlflow.fastai import *
import sys
import time
import os, fnmatch, shutil
import json
import matplotlib

from sklearn.model_selection import train_test_split
# from fastai.text import *
from sklearn.metrics import precision_score, recall_score, accuracy_score
from others.logging import logger
import Preprocess
import torch, gc

sys.path.append("pipeline/functions/DataFunctions")
sys.path.append("../DataFunctions")
from utils import *
import ElasticFunctions as ef
import AzureFunctions as af

ARTIFACTS_DIR = "artifacts"
MODELS_DIR = "models"
LOCAL_AZURE_DIR = "azure_blob"

class Trainer:
    def __init__(self, args):
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Initializing Lesson Summarization...")
        self.run_id = get_runid()
        logger.info("run_id: ", self.run_id)
#         log_param("mode", args.mode)
#         log_param("model_file", args.model_file)
#         log_param("credentials", args.credentials)
        
        # Credentials
        self.credentials = ef.get_credentials(args.credentials)
        
        if args.mode == "train":
            logger.info("TRAIN mode")
            # Get data
            logger.info("Fetching data from Elasticsearch...")
#             ef.updateBaseClassification(self.credentials)
            self.train_df = ef.getIndex(self.credentials, "base-summaries")[["paragraph", "annotationTitle"]]
        elif args.mode == 'predict':
            logger.info("PREDICT mode")
            # Get data
            logger.info("Fetching data from Elasticsearch...")
            self.sentences_df = ef.getSentences(self.credentials)
            self.predict_df = self.sentences_df.replace(True, int(1)).replace(False, int(0))
      
    def preprocess(self, args):
        if args.mode == "train":
            train_folder = f"eva_train_{self.run_id}"
            preprocess_obj = Preprocess.Preprocess(args, self.train_df, train_folder)
            logger.info(f'Converting training data to {train_folder}')
            preprocess_obj.preprocess(args)
#             os.system("python train_dataset_to_stories.py convert <dataset> <train folder name>")
            return train_folder
        elif args.mode == "predict":
            train_folder = f"eva_train_{self.run_id}"
            forecast_folder = f"eva_forecast_{self.run_id}"
            preprocess_obj = Preprocess.Preprocess(args, self.predict_df, forecast_folder)
            logger.info(f'Converting test data to {forecast_folder}')
            preprocess_obj.preprocess(args)
#             os.system("python forecast_dataset_to_stories.py convert <forecasted lessons file> <forecast folder name>")
            return forecast_folder
    
    def run(self, args):
        if args.mode == "train":
            logger.info("Start preprocessing...")
            output_folder = self.preprocess(args)
            logger.info(f"Start training.............")
#             os.system(f"./src/eva_train.sh {args.mode} {self.run_id} {output_folder} {args.run_id}")
        elif args.mode == "predict":
            logger.info("Start preprocessing...")     
            output_folder = self.preprocess(args)
            logger.info("Start predicting...")
#             os.system(f"./src/eva_forecast.sh {args.mode} {self.run_id} {output_folder} ")
#             os.system("./src/eva_forecast.sh <forecast folder name> <train folder name>")

        os.system("chmod +x ./src/eva_run.sh")
        os.system(f"./src/eva_run.sh {args.mode} {self.run_id} {output_folder} {args.run_id}")
        if args.mode == "predict" and args.update_sentences == True:
            self.save_summaries(args, output_folder)
    def save_summaries(self, args, output_folder):
        os.system("python eva_summaries.py args.crdentials output_folder")
    
    
    
    
    
    
    
        