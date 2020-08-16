import subprocess
from mlflow import log_metric, log_param, log_artifact
from os import path
import os
import defaults

def backupIndex(credentials, indexName):
    """Function to log an index ndjson data as an artifact to Azure blob ccontainer
    """
    bashCommand = "elasticdump"
    inputOption = "--input=http://" + credentials["username"] + ":" + credentials["password"] + "@" + credentials["ip_and_port"] + "/" + indexName
    fileName = defaults.DATA_PATH + indexName + ".ndjson"
    if path.exists(fileName):
        os.remove(fileName)
    outputOption = "--output=" + fileName
    subprocess.run([bashCommand, inputOption, outputOption])
    log_artifact(fileName, artifact_path="data")
