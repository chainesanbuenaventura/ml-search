import pandas as pd
import pickle
import seaborn as sns
import mlflow
from mlflow import *
from mlflow.fastai import *
import sys
import time
import os, fnmatch, shutil
import json
import matplotlib

from sklearn.model_selection import train_test_split
from fastai.text import *
from sklearn.metrics import precision_score, recall_score, accuracy_score
from others.logging import logger

sys.path.append("pipeline/functions/DataFunctions")
sys.path.append("../DataFunctions")
from utils import *
import ElasticFunctions as ef
import AzureFunctions as af
import MLFlowFunctions as mf

ARTIFACTS_DIR = "artifacts/data"
MODELS_DIR = "models"
LOCAL_AZURE_DIR = "azure_blob"

LM_EPOCHS = 5
LC_EPOCHS = 50

class Trainer():
    def __init__(self, args):
        logger.info("Initializing Lesson Classification...")
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
#             print("[INFO] Fetching data from Elasticsearch...")
            
            ef.updateBaseClassification(self.credentials)
            self.base_df = ef.getBaseClassification(self.credentials)
            logger.info("Splitting data...")
            self.train_df, self.test_df = train_test_split(self.base_df, test_size=0.2)
            
            # Create ULMFit object
            self.ulmfit_obj = self.UlmFit(self.train_df[["isLesson", "paragraph"]], self.test_df[["isLesson", "paragraph"]])
        elif args.mode == 'predict':
            logger.info("PREDICT mode")
            # Get data
            logger.info("Fetching data from Elasticsearch...")
            self.sentences_df = ef.getSentences(self.credentials)
            self.to_predict_par_df = self.sentences_df[["isLesson", "paragraph"]].replace(True, int(1)).replace(False, int(0))
        
    def train(self, args):
        
        # #  Language model
        self.ulmfit_obj.train_language_model()
        self.ulmfit_obj.save_language_model()
        self.ulmfit_obj.save_lm_file()
        
        # # Lesson classifier
        self.ulmfit_obj.train_text_classification()
        self.ulmfit_obj.save_classifier_file()

        timestamp = get_timestamp()
        plot_losses_filename = f"train_loss_plot-{self.run_id}.jpg"
        self.ulmfit_obj.lesson_learner.recorder.plot_losses(return_fig=True).savefig(os.path.join("artifacts", plot_losses_filename))
        log_artifact(os.path.join(ARTIFACTS_DIR, plot_losses_filename))

        # Evaluate on validation data
        eval_data = "validation"
        valid_forecasts = self.evaluate(eval_data, self.test_df, self.ulmfit_obj.lesson_learner)
        
    def predict(self, args):
        #ROOT_PATH = r"C:\Users\Test Machine\Documents\ADB-CognitiveSearch-ML\pipeline\functions\models"
        ROOT_PATH = "./models"

        # Load saved model file
        #learn = load_learner(Path(ROOT_PATH), "lesson_classif-04-05-2020_11-05-30_PM.pkl")
# #         print("Model dir: ", Path(ROOT_PATH)) 
# #         print("Model file: ", args.model_file)
# #         lesson_learner = load_learner(Path(ROOT_PATH), args.model_file)
#         path, filename = af.downloadArtifact(LOCAL_AZURE_DIR, args.deployment_env, args.run_id)
#         print("Model dir: ", Path(path))
#         print("Model file: ", filename)
#         lesson_learner = load_learner(Path(path), filename)
#         model_uri = af.getModelURI(args.deployment_env, args.run_id)
        model_uri = f"runs:/{args.run_id}/fastai-model"
        lesson_learner = load_model(model_uri)
        
        # Show evaluation results
        eval_data = "test"
        test_forecasts = self.evaluate(eval_data, self.to_predict_par_df, lesson_learner)

        # Get sentences
        # credentials = get_credentials(args.credentials)
        df2 = ef.getSentences(self.credentials)

        # Update isLessons in sentences
        to_predict_par_df2 = self.to_predict_par_df
        to_predict_par_df2.isLesson = test_forecasts

        to_predict_par_df2.isLesson = to_predict_par_df2.isLesson.replace(int(1), True).replace(int(0), False)
        df2.isLesson, df2.paragraph = to_predict_par_df2.isLesson, to_predict_par_df2.paragraph

        if args.update_sentences == True:
            ef.updateSentences(self.scredentials, df2)
            mf.backupIndex(self.credentials, "sentences")
            print(df2.head())
        
    def evaluate(self, eval_data, test_df, lesson_learner):
        logger.info(f"Predicting on {eval_data} dataset...")
        forecasts = []
        ##actual = self.test_df.isLesson.values
        actual = test_df.isLesson.values
        
        #if eval_data == "validation": lesson_learner = self.ulmfit_obj.lesson_learner 

        for p in test_df.paragraph: 
            if eval_data == "predict": print(lesson_learner.predict(p))
            forecasts.append(try_int(lesson_learner.predict(p)[0]))
        
        data = {
                'y_Actual': actual,
                'y_Predicted': forecasts
        }
        precision = precision_score(data['y_Actual'], data['y_Predicted'], average='macro')
        recall = recall_score(data['y_Actual'], data['y_Predicted'], average='macro')
        accuracy = accuracy_score(data['y_Actual'], data['y_Predicted'])
        log_metric(f"{eval_data}_precision", precision)
        log_metric(f"{eval_data}_recall", recall)
        log_metric(f"{eval_data}_accuracy", accuracy)
        logger.info(f"{eval_data} Precision={precision:.4f}".capitalize())
        logger.info(f"{eval_data} Recall={recall:.4f}".capitalize())
        logger.info(f"{eval_data} Accuracy={accuracy:.4f}".capitalize())

        df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
        confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'], margins = True)
        confusion_matrix = confusion_matrix.iloc[:-1,:-1]
        conf_matrix_path = os.path.join(ARTIFACTS_DIR, f"confusion_matrix_RUN{self.run_id}.jpg")
#             pandas_df_to_img(confusion_matrix, conf_matrix_path)
#             log_artifact(conf_matrix_path)

        ##sns.heatmap(confusion_matrix, annot=True)
    
        if eval_data == "predict":
            return forecasts
    
    class UlmFit: 
        def __init__(self, train_df, test_df): 
            
            self.run_id = get_runid()
            self.train_df, self.test_df = train_df, test_df
            self.data_lm =  TextLMDataBunch.from_df(".", train_df, test_df)
            self.data_classifier = TextClasDataBunch.from_df(".", train_df, test_df, 
                                                              vocab=self.data_lm.train_ds.vocab, 
                                                              bs=20)
            self.lm_learner = language_model_learner(self.data_lm, AWD_LSTM, drop_mult=0.5)
            self.lesson_learner = text_classifier_learner(self.data_classifier, 
                                                          AWD_LSTM, drop_mult=0.5, 
                                                          metrics=[accuracy, Precision(), Recall()])
            self.lm_learner.model_dir = "/mlflow/projects/code/models"
            self.lesson_learner.model_dir = "/mlflow/projects/code/models"

        def train_language_model(self):
            print("[INFO] Tuning language model learning rate...")
            self.lm_learner.lr_find()
            timestamp = get_timestamp()
            ##lm_lr_plot_filename = f"lm_lr-{timestamp}.jpg"
            lm_lr_plot_path = os.path.join(ARTIFACTS_DIR, f"lm_lr-RUN{self.run_id}.jpg")
            self.lm_learner.recorder.plot(return_fig=True, suggestion=True).savefig(lm_lr_plot_path)
            log_artifact(lm_lr_plot_path)
            min_grad_lr = self.lm_learner.recorder.min_grad_lr
            log_param("lm_min_grad_lr", min_grad_lr)
            logger.info("Start training language model...")
#             print("[INFO] Start training language model...")
            self.lm_learner.fit_one_cycle(LM_EPOCHS, min_grad_lr)

        def save_language_model(self):
            self.lm_learner.save_encoder('language_model')

        def save_lm_file(self):
            # t = time.localtime()
            # timestamp = time.strftime('%b-%d-%Y_%H%M', t)
            timestamp = get_timestamp()
            ##lm_filename = f"./models/lm-{timestamp}.pkl"
            lm_filename = f"lm-{self.run_id}.pkl"
            lm_path = os.path.join(MODELS_DIR, lm_filename)
            self.lm_learner.export(lm_path)
            logger.info(f"Language model {lm_filename} saved in {MODELS_DIR}")
            log_artifact(lm_path)

        def train_text_classification(self):
            self.lesson_learner.load_encoder('language_model')
            logger.info("Tuning lesson classification model learning rate...")
            self.lesson_learner.lr_find()
            timestamp = get_timestamp()
            ##lc_lr_plot_filename = f"lc_lr-{timestamp}.jpg"
            lc_lr_plot_path = os.path.join(ARTIFACTS_DIR, f"lc_lr-{self.run_id}.jpg")
            self.lesson_learner.recorder.plot(return_fig=True, suggestion=True).savefig(lc_lr_plot_path)
            log_artifact(lc_lr_plot_path)         
            min_grad_lr = self.lesson_learner.recorder.min_grad_lr
            log_param("lc_min_grad_lr", min_grad_lr)
            logger.info("Start training lesson classification model...")
            self.lesson_learner.fit_one_cycle(LC_EPOCHS, min_grad_lr)

        def save_classifier_file(self):
            # t = time.localtime()
            # timestamp = time.strftime('%b-%d-%Y_%H%M', t)
            timestamp = get_timestamp()
            lc_filename = f"lesson_classif-{self.run_id}.pkl"
            lc_path = os.path.join(MODELS_DIR, lc_filename)
            self.lesson_learner.export(lc_path)
            logger.info(f"Classification model {lc_filename} saved in {MODELS_DIR}"
#             log_artifact(lc_path)
            log_model(self.lesson_learner, artifact_path="fastai-model")