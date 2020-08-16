import mlflow
import json
import os
from os import path
import time
import pandas as pd

import defaults

# import matplotlib.pyplot as plt
# from pandas.table.plotting import table # EDIT: see deprecation warnings below

# # MLflow Tracking functions
# def log_param(param_name, value):
#     print(f"[INFO] Logging parameter: {param_name}")
#     mlflow.log_param(param_name, value)
    
# def log_metric(metric_name, value, step=None):
#     print(f"[INFO] Logging metric: {metric_name}")
#     mlflow.log_metric(metric_name, value, step=None)
    
# def log_artifact(source_dir, artifact_path=None):
#     print(f"[INFO] Logging artifacts in {source_dir}...")
#     mlflow.log_artifacts(local_dir, artifact_path=None)

# def pandas_df_to_img(df, img_path):

#     ax = plt.subplot(111, frame_on=False) # no visible frame
#     ax.xaxis.set_visible(False)  # hide the x axis
#     ax.yaxis.set_visible(False)  # hide the y axis

#     table(ax, df)  # where df is your data frame

#     plt.savefig(img_path)

def get_credentials(filename):
    """Load credentials from JSON file
    """
    CREDENTIALS_ROOT = "../credentials"
    with open(os.path.join(CREDENTIALS_ROOT, filename)) as f:
        data = json.load(f)
    return data

def get_runid():
    run_id = mlflow.active_run().info.run_id
    return run_id

def get_timestamp():
    os.environ['TZ'] = 'Singapore'
    time.tzset()
    timestamp = time.strftime("%Y%m%d-%H.%M.%S")
    return timestamp

def get_tracking_uri():
    with open(os.path.join(defaults.CREDENTIALS_PATH, "trackinguri"), "r") as f:
        data = f.readline()
    return data

def get_storage_key():
    if path.exists(os.path.join(defaults.CREDENTIALS_PATH, "storagekey")):
        with open(os.path.join(defaults.CREDENTIALS_PATH, "storagekey"), "r") as f:
            data = f.readline()
    elif path.exists(os.path.join(defaults.CREDENTIALS_PATH_2, "storagekey")):
        with open(os.path.join(defaults.CREDENTIALS_PATH_2, "storagekey"), "r") as f:
            data = f.readline()
    else:
        with open(os.path.join(defaults.CREDENTIALS_PATH_3, "storagekey"), "r") as f:
            data = f.readline()
    return data

def get_connection_string():
    if path.exists(os.path.join(defaults.CREDENTIALS_PATH, "connectionstring")):
        with open(os.path.join(defaults.CREDENTIALS_PATH, "connectionstring"), "r") as f:
            data = f.readline()
    elif path.exists(os.path.join(defaults.CREDENTIALS_PATH_2, "connectionstring")):
        with open(os.path.join(defaults.CREDENTIALS_PATH_2, "connectionstring"), "r") as f:
            data = f.readline()
    else:
        with open(os.path.join(defaults.CREDENTIALS_PATH_3, "connectionstring"), "r") as f:
            data = f.readline()
    return data


# def setup_mlflow(experiment_name):
#     print("MLflow Version:", mlflow.version.VERSION)
#     mlflow.set_tracking_uri("http://40.112.217.252:5000/")
# #     mlflow.set_tracking_uri("/mlruns")
#     print("Tracking URI:", mlflow.tracking.get_tracking_uri())

#     experiment_name = "dev-LessonsClassification"
#     print("experiment_name: ", experiment_name)
#     mlflow.set_experiment(experiment_name)

#     client = mlflow.tracking.MlflowClient()
#     experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
#     print("experiment_id: ", experiment_id)
