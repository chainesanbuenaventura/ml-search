import defaults
import sys
sys.path.append("pipeline/functions/DataFunctions")
sys.path.append("../DataFunctions")
from utils import *

from azure.storage.blob import BlobServiceClient
import os

CONNECTION_STRING = get_connection_string()
STORAGE_ACCOUNT_KEY = get_storage_key()
AZURE_CONTAINER = defaults.MLFLOW_CONTAINER

def ls_files(client, path, recursive=False):
    """List files under a path, optionally recursively
    """
    if not path == '' and not path.endswith('/'):
      path += '/'

    blob_iter = client.list_blobs(name_starts_with=path)
    files = []
    for blob in blob_iter:
      relative_path = os.path.relpath(blob.name, path)
      if recursive or not '/' in relative_path:
        files.append(relative_path)
    return files

def get_blobs(connection_string):
    """Get list of full path of blobs inside a container
    """
    # Connection string for the storage account.
    #connection_string = CONNECTION_STRING

    # Name of the container you are interested in.
    #container_name = 'mlflow-container'

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    client = blob_service_client.get_container_client(AZURE_CONTAINER)

    files = ls_files(client, '', recursive=True)
    
    return files

def getModelURI(deployment_env, run_id):
    """Return AzureBlob URI of trained model
    """
    assert deployment_env in ["staging", "development", "production"]    
        
    files = get_blobs(CONNECTION_STRING)
    
    keywords = [deployment_env, run_id, "lesson_classif", ".pkl"]
    for file in files:
        if len([keyword for keyword in keywords if keyword in file]) == len(keywords):
            model_uri = file
            return model_uri

def downloadArtifact(local_dir, deployment_env, run_id):
    """Download artifact in Azure Storage Blob to local
    """
    assert deployment_env in ["staging", "development", "production"]
      
    model_uri = getModelURI(deployment_env, run_id)
    model_uri_dirs = model_uri.split("/")
    
    path = f"azure_blob/models/{deployment_env}/{run_id}"
    
    if os.path.exists(path) == False:
        os.makedirs(path)

    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER, blob=model_uri) 
    
    filename = model_uri_dirs[-1]
    blob_path = os.path.join(path, filename)
    with open(blob_path, "wb") as downloadFile:
        downloadFile.write(blob_client.download_blob().readall())
        
    return path, filename

def savePCRPDF(PCRFilePath, PCRFileName):
    """Save downloaded PCR pdf to Azure blob container
    """
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container="pcrs", blob=PCRFileName)

    with open(PCRFilePath, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

def downloadPCRPDF(folderName, PCRFileName):
    """Download PCR pdf from Azure blob container
    """
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container="pcrs", blob=PCRFileName)

    with open(folderName + PCRFileName, "wb") as downloadFile:
        downloadFile.write(blob_client.download_blob().readall())

def savePCRDOCX(PCRFilePath, PCRFileName):
    """Save downloaded PCR docx to Azure blob container
    """
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container="pcrs-docx", blob=PCRFileName)

    with open(PCRFilePath, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

def downloadPCRDOCX(folderName, PCRDOCXFileName):
    """Download PCR docx from Azure blob container
    """
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container="pcrs-docx", blob=PCRDOCXFileName)

    with open(folderName + PCRDOCXFileName, "wb") as downloadFile:
        downloadFile.write(blob_client.download_blob().readall())

def docxExists(fileName):
    """Checks if a docx file exist in Azure blob container 
    """
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container="pcrs-docx", blob=fileName)
    try:
        blob_client.download_blob().readall()
        return True
    except:
        return False

def getDocxList(fileName):
    """Get all the docx file names from Azure blob container of a given file name
    """
    docxList = []
    i = 0
    while(True):
        docxFileName = fileName + "_Page" + str(i) + "_PdfToWord.docx"
        print(docxFileName)
        if docxExists(docxFileName):
            docxList.append(docxFileName)
        else:
            break
        i = i + 1
    return docxList

def downloadLDAModel(args, modelFilePath):
    """Download an LDA model stored in a run's artifact storage in Azure
    """
    modelFile = args.environment + "/lessonsClustering/" + args.run_id_model + "/artifacts/models/lda.model"
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=defaults.MLFLOW_CONTAINER, blob=modelFile)

    with open(modelFilePath, "wb") as downloadFile:
        downloadFile.write(blob_client.download_blob().readall())

def downloadDataFile(args):
    """Download an ndjson file of an index stored in a run's artifact storage in Azure
    """
    dataFile = args.environment + "/" + args.module + "/" + args.run_id + "/artifacts/data/" + args.index_name + ".ndjson"
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=defaults.MLFLOW_CONTAINER, blob=dataFile)

    try:
        with open(defaults.DATA_PATH + args.index_name + ".ndjson", "wb") as downloadFile:
            downloadFile.write(blob_client.download_blob().readall())
        return True
    except:
        print("Data not found. Check if the index name, module name, or run id provided exists")
        return False

def saveFile(filePath, fileName, containerName):
    """Save a file to Azure blob container
    """
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=containerName, blob=fileName)

    with open(filePath + fileName, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

def downloadFile(fileName, containerName):
    """Download a file from Azure blob container
    """
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=containerName, blob=fileName)

    with open(fileName, "wb") as downloadFile:
        downloadFile.write(blob_client.download_blob().readall())