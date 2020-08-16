import pandas as pd
import pickle
import seaborn as sns
import mlflow
import sys
import time
import os, fnmatch, shutil

##from fuzzywuzzy import fuzz
from sklearn.model_selection import train_test_split
from fastai.text import *
from sklearn.metrics import precision_score, recall_score, accuracy_score
# from datetime import *
from DataFunctions import ElasticFunctions as ef
sys.path.append("pipeline/functions/DataFunctions")
sys.path.append("../DataFunctions")
from utils import *


class UlmFit: 
    def __init__(self, train_df, test_df): 
        self.data_lm =  TextLMDataBunch.from_df(".", train_df, test_df)
        self.data_classifier = TextClasDataBunch.from_df(".", train_df, test_df, 
                                                          vocab=self.data_lm.train_ds.vocab, 
                                                          bs=20)
        self.lm_learner = language_model_learner(self.data_lm, AWD_LSTM, drop_mult=0.5)
        self.lesson_learner = text_classifier_learner(self.data_classifier, 
                                                      AWD_LSTM, drop_mult=0.5, 
                                                      metrics=[accuracy, Precision(), Recall()]).to_fp16()
        
    def train_language_model(self):
        print("[INFO] Training language model...")
        self.lm_learner.lr_find()
        self.lm_learner.recorder.plot(suggestion=True)
        min_grad_lr = self.lm_learner.recorder.min_grad_lr
        self.lm_learner.fit_one_cycle(5, min_grad_lr)
    
    def save_language_model(self):
        self.lm_learner.save_encoder('language_model')
    
    def save_lm_file(self):
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M', t)
        self.lm_learner.export(f"./models/lm-{timestamp}.pkl")
        print(f"[INFO] Language model ../models/lm-{timestamp}.pkl saved")
    
    def train_text_classification(self):
        print("[INFO] Training lesson classification model...")
        self.lesson_learner.load_encoder('language_model')
        self.lesson_learner.lr_find()
        self.lesson_learner.recorder.plot(suggestion=True)
        min_grad_lr = self.lesson_learner.recorder.min_grad_lr
        self.lesson_learner.fit_one_cycle(50, min_grad_lr)

    def save_classifier_file(self):
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M', t)
        self.lesson_learner.export(f"./models/lesson_classif-{timestamp}.pkl")
        print(f"[INFO] Classification model ./models/lesson_classif-{timestamp}.pkl saved")

    def evaluate(self):
#         test_lesson_par_df = pd.read_csv(os.path.join(data_dir, test_data_file))

        forecasts = []
        actual = test_lesson_par_df.is_lesson.values

        for p in test_lesson_par_df.paragraph: 
        #     print(learn_classif.predict(p))
            forecasts.append(try_int(self.lesson_learner.predict(p)[0]))
        
        data = {
                'y_Actual': actual,
                'y_Predicted': forecasts
        }
        precision = precision_score(y_actual,y_pred, average='macro')
        recall = recall_score(y_actual,y_pred, average='macro')
        accuracy = accuracy_score(y_actual, y_pred)
        log_metric("precision", precision)
        log_metric("recall", recall)
        log_metric("accuracy", accuracy)
        
        print(f"Precision={precision}")
        print(f"Recall={recall}")
        print(f"Accuracy={accuracy}")

        df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
        confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'], margins = True)
        confusion_matrix = confusion_matrix.iloc[:-1,:-1]

        ##sns.heatmap(confusion_matrix, annot=True)


class ElasticsearchIO:
    import time
    
    def __init__(self, creds_file):
        self.credentials = get_credentials(creds_file)
        ef.updateBaseClassification(self.credentials)
        self.df = ef.getBaseClassification(self.credentials)
        print(self.df.shape)

    def get_for_predict_dataframe(self):

        """credentials = {
            # "ip_and_port": "52.163.240.214:9200",
            "ip_and_port": "52.230.8.63:9200",
            "username": "elastic",
            "password": "Welcometoerni!"
        }

        prodCredentials = {
            "ip_and_port": "52.163.240.214:9200",
            # "ip_and_port": "52.230.8.63:9200",
            "username": "elastic",
            "password": "Welcometoerni!"
        }"""

        ##credentials = get_credentials(args.credentials)

#         # Get lessons data from database
#         df = ef.getBaseClassification(self.credentials)

        to_predict_par_df = self.df[["isLesson", "paragraph"]].replace(True, int(1)).replace(False, int(0))

        return to_predict_par_df

    def get_train_test(self):

        """credentials = {
            # "ip_and_port": "52.163.240.214:9200",
            "ip_and_port": "52.230.8.63:9200",
            "username": "elastic",
            "password": "Welcometoerni!"
        }

        prodCredentials = {
            "ip_and_port": "52.163.240.214:9200",
            # "ip_and_port": "52.230.8.63:9200",
            "username": "elastic",
            "password": "Welcometoerni!"
        }"""

        ##credentials = get_credentials(args.credentials)

        # Get lessons data from database
#         df = ef.getBaseClassification(self.credentials)

        train_df, test_df = train_test_split(self.df, test_size=0.2)
        
        return train_df, test_df

def setup_mlflow():
    print("MLflow Version:", mlflow.version.VERSION)
    print("Tracking URI:", mlflow.tracking.get_tracking_uri())

    experiment_name = "lesson-classif-windows"
    print("experiment_name:",experiment_name)
    mlflow.set_experiment(experiment_name)

    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    print("experiment_id:",experiment_id)

#     now = int(time.time()+.5)    


def main():
    
    setup_mlflow()

    es_io = ElasticsearchIO(creds_file)

    train_df, test_df = es_io.get_train_test()

    # UlmFit
    ulmfit_obj = UlmFit(train_df, test_df)
    
    #  Language model
    ulmfit_obj.train_language_model()
    ulmfit_obj.save_language_model()
    lm_path = ulmfit_obj.save_lm_file()
    
    # Lesson classifier
    ulmfit_obj.train_text_classification()
    lesson_classif_path = ulmfit_obj.save_lm_file()

    ##ulmfit_obj.lesson_learner.recorder.plot_losses()

    # Show evaluation results
    ulmfit_obj.evaluate()

    
if __name__ == "__main__":
    # Arguments
    parser = ArgumentParser()
    parser.add_argument("--creds_file", dest="creds_file", default="stagingcredentials.json", required=True, action='store_true')
    args = parser.parse_args()
    main()
    
    


    
