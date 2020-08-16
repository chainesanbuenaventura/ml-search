from argparse import ArgumentParser
import sys
import os
import pandas as pd
import numpy as np
import mlflow
from mlflow import log_metric, log_param, log_artifact

from ParagraphExtraction import PCRParagraphExtractor
sys.path.append("pipeline/functions/DataFunctions")
sys.path.append("../DataFunctions")
from utils import *
import defaults
import ElasticFunctions as ef
import AzureFunctions as af
import MLFlowFunctions as mf

if __name__ == "__main__":

    tracking_uri = get_tracking_uri()

# Arguments
    parser = ArgumentParser()
    parser.add_argument("--environment", dest="environment", default="development", required=True, help='Set which environment to run the paragraph extraction (development, staging, or production)')
    args = parser.parse_args()

    # docx_path = "../data/raw_docx/"
    # pcr_data = pd.read_excel("./data/pcr_data_3.xlsx", index_col=0)
    # pcr_data['year'] = [month_year[1] for month_year in pcr_data['Month Year'].str.split(' ')]

    credentials = ef.get_credentials("localcredentials.json")
    if args.environment == "staging":
        credentials = ef.get_credentials("stagingcredentials.json")
    elif args.environment == "production":
        credentials = ef.get_credentials("prodcredentials.json")
    
    experiment_name = "dev-ParagraphExtraction"
    if args.environment == "production":
        experiment_name = "ParagraphExtraction"
    elif args.environment == "staging":
        experiment_name = "staging-ParagraphExtraction"
    mlflow.set_experiment(experiment_name)
    client = mlflow.tracking.MlflowClient()

    with mlflow.start_run():
        log_param("environment", args.environment)
        folderName = defaults.DATA_PATH
        PCRsDF = ef.getIndex(credentials, "pcrs")

    # Extract from docx files
        for i, r in PCRsDF.iterrows():
            if r["isExtracted"] == False and r["tentative"] == False:
                extractedParagraphs = pd.DataFrame()
                projectNumbers = []
                titles = []
                allParagraphs = []
                themes = []
                sectors = []
                countries = []
                modalities = []
                month = []
                year = []
                urlToFile = []
                allMainHeaders = []
                allSubHeaders = []

                i = 0
                while(True):
                    docxFileName = r["fileName"] + "_Page" + str(i) + "_PdfToWord.docx"
                    if af.docxExists(docxFileName):
                        if r["downloadLink"]:
                            print("Extracting from: " + r["downloadLink"])
                        else:
                            print("Extracting from: " + r["title"])
                        af.downloadPCRDOCX(folderName, docxFileName)
                        pcr_extractor = PCRParagraphExtractor(folderName + docxFileName)
                        paragraphs, main_headers, sub_headers = pcr_extractor.extract_paragraphs()
                        if len(paragraphs):
                            projectNumbers = projectNumbers + [r["projectNumber"]] * len(paragraphs)
                            titles = titles + [r["title"]] * len(paragraphs)
                            themes = themes + [r["themes"]] * len(paragraphs)
                            sectors = sectors + [r["sectors"]] * len(paragraphs)
                            countries = countries + [r["countries"]] * len(paragraphs)
                            modalities = modalities + [r["modalitiesFromWebsite"]] * len(paragraphs)
                            month = month + [r["month"]] * len(paragraphs)
                            year = year + [r["year"]] * len(paragraphs)
                            urlToFile = urlToFile + [r["downloadLink"]] * len(paragraphs)
                            allParagraphs = allParagraphs + paragraphs
                            allMainHeaders = allMainHeaders + main_headers
                            allSubHeaders = allSubHeaders + sub_headers
                        os.remove(folderName + docxFileName)
                    else:
                        if r["downloadLink"]:
                            print("No docx in Azure storage: " + r["downloadLink"])
                        else:
                            print("No docx in Azure storage: " + r["title"])
                        break
                    i = i + 1

                extractedParagraphs["projectNumber"] = projectNumbers
                extractedParagraphs["title"] = titles
                extractedParagraphs["themes"] = themes
                extractedParagraphs["sectors"] = sectors
                extractedParagraphs["countries"] = countries
                extractedParagraphs["modalities"] = modalities
                extractedParagraphs["month"] = month
                extractedParagraphs["year"] = year
                extractedParagraphs["urlToFile"] = urlToFile
                extractedParagraphs["paragraph"] = allParagraphs
                extractedParagraphs["mainHeader"] = allMainHeaders
                extractedParagraphs["subHeader"] = allSubHeaders

    # Add new sentences to elasticsearch
                ef.addNewSentences(credentials, extractedParagraphs)
            
    # Mark PCR as extracted
                ef.updateIsExtracted(credentials, r, True)


    # Backup pcrs index
        mf.backupIndex(credentials, "pcrs")

    # Backup sentences index
        mf.backupIndex(credentials, "sentences")

    # Unused code after refactoring
"""
        import glob
        pcr_data['year'].astype(int)
        project_numbers = (set(pcr_data['projectNumber'].values) 
                           - set(pcr_data.groupby('projectNumber').count().query('title > 1').index.values))
        project_number = list(project_numbers)[0]
        docx_name = glob.glob(docx_path + f"*{project_number}*")
        pcr_extractor = PCRParagraphExtractor(docx_name[0])
        paragraphs, main_headers, sub_headers = pcr_extractor.extract_paragraphs()

        project_numbers = (set(pcr_data['projectNumber'].values) 
                           - set(pcr_data.groupby('projectNumber').count().query('title > 1').index.values))
        for index in project_numbers:
            docx_name = glob.glob(docx_path + f"*{project_number}*")
            try:
                pcr_extractor = PCRParagraphExtractor(credentials, docx_name[0])
                paragraphs, main_headers, sub_headers = pcr_extractor.extract_paragraphs()

                extracted_par_data.append(np.array([[proj_number] * len(paragraphs),
                                                    paragraphs, 
                                                    main_headers, 
                                                    sub_headers]))
            except:
                continue
                # print(f"No Document found for {project_number}")
"""