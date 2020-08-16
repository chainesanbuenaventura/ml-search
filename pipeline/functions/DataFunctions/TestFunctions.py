import pandas as pd
import os
import shutil
from elasticsearch import Elasticsearch, helpers
from pandasticsearch import Select
from es_pandas import es_pandas
import os.path

import defaults
from utils import *
import ElasticFunctions as ef
import ExtractionFunctions as exf
import AzureFunctions as af
import MLFlowFunctions as mf

"""Contains functions for data fixes and tests
"""

def saveTFIDF(credentials, dfTFIDF):
    ep = es_pandas('http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"])
    ep.init_es_tmpl(dfTFIDF, "tfidf")
    ep.to_es(dfTFIDF, "tfidf", doc_type="tfidf")

def getTFIDF(credentials):
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    doc = {
            'size' : 10000,
            'query': {
                'match_all' : {}
        }
    }
    dfTFIDF = pd.DataFrame()
    data = es.search(index="tfidf", body=doc, scroll='1m')
    scrollId = data['_scroll_id']
    scrollSize = len(data['hits']['hits'])
    while scrollSize > 0:
        if dfTFIDF.empty:
            dfTFIDF = Select.from_dict(data).to_pandas()
        else:
            dfTFIDF = dfTFIDF.append(Select.from_dict(data).to_pandas())
        data = es.scroll(scroll_id = scrollId, scroll = '1m')
        scrollId = data['_scroll_id']
        scrollSize = len(data['hits']['hits'])
    return dfTFIDF

def saveBaseClassifications(credentials):
    """
    Description: Function to initially save the base classifications
    Returns: None
    """
    baseDF = pd.read_csv("data/base_classification_data.csv")
    baseDF = baseDF.rename(columns={"is_lesson": "isLesson"})
    baseDF["source"] = "base"
    baseDF["sentencesId"] = ""
    baseDF["isLesson"] = baseDF["isLesson"]
    ep = es_pandas('http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"])
    ep.init_es_tmpl(baseDF, "base-classification")
    ep.to_es(baseDF, "base-classification", doc_type="base-classification")

def saveBaseSummaries(credentials):
    """
    Description: Function to initially save the base summaries
    Returns: None
    """
    baseDF = pd.read_csv(defaults.DATA_PATH + "train_data_lesson_title.csv")
    baseDF = baseDF.rename(columns={"human generated title": "annotationTitle"})
    baseDF = baseDF[baseDF["paragraph"] != '"']
    baseDF = baseDF.dropna()
    baseDF["source"] = "base"
    baseDF["sentencesId"] = ""
    ep = es_pandas('http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"])
    ep.init_es_tmpl(baseDF, "base-summaries")
    ep.to_es(baseDF, "base-summaries", doc_type="base-summaries")

def computeIds(credentials):
    sentencesDF = ef.getSentences(credentials)
    sentencesDF["id"] = sentencesDF["_id"]
    sentencesDF["id"] = sentencesDF["id"].astype('str')
    ef.deleteIndex(credentials, "sentences")
    if "_index" in sentencesDF.columns:
        sentencesDF = sentencesDF.drop(columns=["_index"])
    if "_type" in sentencesDF.columns:
        sentencesDF = sentencesDF.drop(columns=["_type"])
    if "_id" in sentencesDF.columns:
        sentencesDF = sentencesDF.drop(columns=["_id"])
    if "_score" in sentencesDF.columns:
        sentencesDF = sentencesDF.drop(columns=["_score"])
    ep = es_pandas('http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"])
    ep.init_es_tmpl(sentencesDF, "sentences")
    ep.to_es(sentencesDF, "sentences", doc_type="sentences")

def resetColumns(credentials, indexName):
    sentencesDF = ef.getIndex(credentials, indexName)
    # sentencesDF["relatedLessons"] = [ [] for i in range(len(sentencesDF)) ]
    # sentencesDF["topTopics"] = [ [] for i in range(len(sentencesDF)) ]
    # sentencesDF["topic"] = -1
    # sentencesDF["topic"] = sentencesDF["topic"].astype('long')
    # for i, r in sentencesDF.iterrows():
    #     sentencesDF.at[i, "year"] = int(r["year"]) if not r["year"] == [] else None
    # sentencesDF["year"] = sentencesDF["year"].astype('long')
    # sentencesDF["lessonStrength"] = -1
    sentencesDF["lessonStrength"] = sentencesDF["lessonStrength"].astype('float')
    # sentencesDF["sectorDiscriminator"] = sentencesDF["sectorDiscriminator"].astype('long')
    actions = [
        {
            "_index": indexName,
            "_id": row["_id"],
            "_source": {
                "referenceId": str(row["referenceId"]).zfill(20),
                "title": row["title"],
                "paragraph": row["paragraph"],
                "themes": row["themes"],
                "sectors": row["sectors"],
                "sectorDiscriminator": row["sectorDiscriminator"],
                "countries": row["countries"],
                "lessonStrength": row["lessonStrength"],
                "relatedLessons": row["relatedLessons"],
                "topic": row["topic"],
                "topTopics": row["topTopics"],
                "modalities": row["modalities"],
                "lessonType": row["lessonType"],
                "isLesson": row["isLesson"], 
                "month": row["month"], 
                "year": row["year"], 
                "annotationTitle": row["annotationTitle"],
                "summary": row["summary"],
                "context":  row["context"],
                "annotatedBy": row["annotatedBy"],
                "annotationSummary": row["annotationSummary"],
                "lastAnnotated": row["lastAnnotated"],
                "annotationStatus": row["annotationStatus"],
                "urlToFile": row["urlToFile"],
                "lastUpdated": "",
                "source": row["source"]
            }
        }
        for i, row  in sentencesDF.iterrows()
    ]
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    helpers.bulk(es, actions)

def resetRelatedLessons(credentials, indexName):
    sentencesDF = ef.getIndex(credentials, indexName)
    # sentencesDF["relatedLessons"] = [ "" for i in range(len(sentencesDF)) ]
    # sentencesDF["topTopics"] = [ [] for i in range(len(sentencesDF)) ]
    # sentencesDF["topic"] = -1
    # sentencesDF["topic"] = sentencesDF["topic"].astype('long')
    # for i, r in sentencesDF.iterrows():
    #     sentencesDF.at[i, "year"] = int(r["year"]) if not r["year"] == [] else None
    # sentencesDF["year"] = sentencesDF["year"].astype('long')
    # # sentencesDF["lessonStrength"] = -1
    # sentencesDF["lessonStrength"] = sentencesDF["lessonStrength"].astype('float')
    # sentencesDF["sectorDiscriminator"] = sentencesDF["sectorDiscriminator"].astype('long')

    def getRelatedLessons(relatedLessons):
        if relatedLessons == "":
            return []
        return relatedLessons

    actions = [
        {
            "_index": indexName,
            "_id": row["_id"],
            "_source": {
                "referenceId": str(row["referenceId"]).zfill(20),
                "title": row["title"],
                "paragraph": row["paragraph"],
                "themes": row["themes"],
                "sectors": row["sectors"],
                "sectorDiscriminator": row["sectorDiscriminator"],
                "countries": row["countries"],
                "lessonStrength": row["lessonStrength"],
                "relatedLessons": getRelatedLessons(row["relatedLessons"]),
                "topic": row["topic"],
                "topTopics": row["topTopics"],
                "modalities": row["modalities"],
                "lessonType": row["lessonType"],
                "isLesson": row["isLesson"], 
                "month": row["month"], 
                "year": row["year"], 
                "annotationTitle": row["annotationTitle"],
                "summary": row["summary"],
                "context":  row["context"],
                "annotatedBy": row["annotatedBy"],
                "annotationSummary": row["annotationSummary"],
                "lastAnnotated": row["lastAnnotated"],
                "annotationStatus": row["annotationStatus"],
                "urlToFile": row["urlToFile"],
                "lastUpdated": "",
                "source": row["source"]
            }
        }
        for i, row  in sentencesDF.iterrows()
    ]
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    helpers.bulk(es, actions)

def recoverProjectNumbers(credentials):
    sentencesDF = ef.getIndex(credentials, "sentences")
    s = pd.read_excel("data/sentences-20200508.xlsx")
    projectNumbers = s["Project Number"].tolist()
    referenceIds = s["reference_id"].tolist()

    def getProjectNumber(referenceId):
        return projectNumbers[referenceIds.index(int(referenceId))]

    actions = [
        {
            "_index": "sentences",
            "_id": row["_id"],
            "_source": {
                "referenceId": str(row["referenceId"]).zfill(20),
                "title": row["title"],
                "projectNumber": getProjectNumber(row["referenceId"]),
                "paragraph": row["paragraph"],
                "themes": row["themes"],
                "sectors": row["sectors"],
                "sectorDiscriminator": row["sectorDiscriminator"],
                "countries": row["countries"],
                "lessonStrength": row["lessonStrength"],
                "relatedLessons": row["relatedLessons"],
                "topic": row["topic"],
                "topTopics": row["topTopics"],
                "modalities": row["modalities"],
                "lessonType": row["lessonType"],
                "isLesson": row["isLesson"], 
                "month": row["month"], 
                "year": row["year"], 
                "annotationTitle": row["annotationTitle"],
                "summary": row["summary"],
                "context":  row["context"],
                "annotatedBy": row["annotatedBy"],
                "annotationSummary": row["annotationSummary"],
                "lastAnnotated": row["lastAnnotated"],
                "annotationStatus": row["annotationStatus"],
                "urlToFile": row["urlToFile"],
                "lastUpdated": row["lastUpdated"],
                "source": row["source"]
            }
        }
        for i, row  in sentencesDF.iterrows()
    ]
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    helpers.bulk(es, actions)

def updatePCRIds(credentials):
    sentencesDF = ef.getIndex(credentials, "sentences")
    PCRsDF = ef.getIndex(credentials, "pcrs")
    projectNumbers = PCRsDF["projectNumber"].tolist()
    ids = PCRsDF["_id"].tolist()

    def getPCRId(projectNumber):
        if projectNumber in projectNumbers:
            return ids[projectNumbers.index(projectNumber)]
        return ""

    actions = [
        {
            "_index": "sentences",
            "_id": row["_id"],
            "_source": {
                "referenceId": str(row["referenceId"]).zfill(20),
                "title": row["title"],
                "projectNumber": row["projectNumber"],
                "PCRId": getPCRId(row["projectNumber"]),
                "paragraph": row["paragraph"],
                "themes": row["themes"],
                "sectors": row["sectors"],
                "sectorDiscriminator": row["sectorDiscriminator"],
                "countries": row["countries"],
                "lessonStrength": row["lessonStrength"],
                "relatedLessons": row["relatedLessons"],
                "topic": row["topic"],
                "topTopics": row["topTopics"],
                "modalities": row["modalities"],
                "lessonType": row["lessonType"],
                "isLesson": row["isLesson"], 
                "month": row["month"], 
                "year": row["year"], 
                "annotationTitle": row["annotationTitle"],
                "summary": row["summary"],
                "context":  row["context"],
                "annotatedBy": row["annotatedBy"],
                "annotationSummary": row["annotationSummary"],
                "lastAnnotated": row["lastAnnotated"],
                "annotationStatus": row["annotationStatus"],
                "urlToFile": row["urlToFile"],
                "lastUpdated": row["lastUpdated"],
                "source": row["source"]
            }
        }
        for i, row  in sentencesDF.iterrows()
    ]
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    helpers.bulk(es, actions)

def producePCRs(credentials):
    sentencesDF = ef.getIndex(credentials, "sentences")
    PCRsDF = ef.getIndex(credentials, "pcrs")
    projectNumbers = PCRsDF["projectNumber"].tolist()

    for i, r in sentencesDF.iterrows():
        if r["PCRId"] == "" and not r["projectNumber"] in projectNumbers:
            pcr = {}
            pcr["projectNumber"] = r["projectNumber"]
            pcr["title"] = r["title"]
            pcr["sectors"] = r["sectors"]
            pcr["countries"] = r["countries"]
            pcr["themes"] = r["themes"]
            pcr["downloadLink"] = r["urlToFile"]
            pcr["fileName"] = ""
            pcr["monthYear"] = str(r["month"]) + " " + str(r["year"])
            pcr["month"] = r["month"]
            pcr["year"] = r["year"]
            pcr["milestoneApprovalDate"] = ""
            pcr["milestoneEffectivityDate"] = ""
            pcr["milestoneSigningDate"] = ""
            pcr["safeguardCategories"] = ""
            pcr["sourceOfFunding"] = ""
            pcr["modalitiesFromDump"] = ""
            pcr["modalitiesFromWebsite"] = r["modalities"]
            pcr["uniqueModalitiesFromDump"] = ""
            ef.savePCR(credentials, pcr)
            projectNumbers.append(r["projectNumber"])

def saveIndex(credentials, df, indexName):
    if "_index" in df.columns:
        df = df.drop(columns=["_index"])
    if "_type" in df.columns:
        df = df.drop(columns=["_type"])
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])
    if "_score" in df.columns:
        df = df.drop(columns=["_score"])
    ep = es_pandas('http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"])
    ep.init_es_tmpl(df, indexName)
    ep.to_es(df, indexName, doc_type=indexName)
    
def downloadIndex(credentials, indexName):
    indexDF = ef.getIndex(credentials, indexName)
    indexDF.to_excel(defaults.DATA_PATH + "index.xlsx")

def recomputeFileNames(credentials):
    PCRsDF = ef.getIndex(credentials, "pcrs")
    fileNames = []
    for d in PCRsDF["downloadLink"].tolist():
        if not d:
            fileNames.append(None)
        else:
            startFileName = 0
            for i in range(len(d)-1, 0, -1):
                if d[i] == '/':
                    startFileName = i + 1
                    break
            fileNames.append(d[startFileName:])
    PCRsDF["fileName"] = fileNames
    ef.updatePCRs(credentials, PCRsDF)

def renameDocx():
    newFileNames = []
    oldPCRData = pd.read_excel("data/pcr_data_3.xlsx")
    for d in oldPCRData["Download Link"].tolist():
        startFileName = 0
        if not d == d:
            newFileNames.append(None)
        else:
            for i in range(len(d)-1, 0, -1):
                if d[i] == '/':
                    startFileName = i + 1
                    break
            newFileNames.append(d[startFileName:])
    oldPCRData["fileName"] = newFileNames
    oldFileNames = oldPCRData["File Names"].tolist()
    suffix = ".pdf_Page"
    for path, subdirs, files in os.walk("data/raw_docx/DOCX/"):
        for name in files:
            docxFileName = os.path.join(path, name)
            if not docxFileName[-5:] == ".docx":
                continue
            newFileName = None
            fileNameFound = None
            for f in oldFileNames:
                modifiedFileName = f[f.find("-")+1:]
                if modifiedFileName in docxFileName:
                    fileNameFound = f
                    newFileName = "data/docx/" + newFileNames[oldFileNames.index(f)] + docxFileName[docxFileName.find("_Page"):]
                    print(docxFileName)
                    print(newFileName)
                    print("\n")
                    shutil.move(docxFileName, newFileName)
                    break

def saveAllPCRDOCX():
    for path, subdirs, files in os.walk("data/docx/"):
        for name in files:
            docxFileName = os.path.join(path, name)
            print(docxFileName + "\n" + name)
            af.savePCRDOCX(docxFileName, name)

def countCurrentPCRs(credentials):
    sentencesDF = ef.getIndex(credentials, "sentences")
    PCRsDF = ef.getIndex(credentials, "pcrs")
    projectNumbers = PCRsDF["projectNumber"].tolist()
    projectNumbersFound = []
    print(len(projectNumbers))
    for i, r in sentencesDF.iterrows():
        if r["projectNumber"] in projectNumbers:
            projectNumbersFound.append(r["projectNumber"])
    print(len(list(set(projectNumbersFound))))

def setIsExtracted(fileName):
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    doc = {
        "query": {
         "match_phrase": {
            "fileName": fileName
          }
      }
    }
    data = es.search(index="pcrs", body=doc)
    df = Select.from_dict(data).to_pandas()
    action = [
        {
            "_index": "pcrs",
            "_id": row["_id"],
            "_source": {
                "projectNumber": row["projectNumber"],
                "isExtracted": False,
                "tentative": row["tentative"],
                "title": row["title"],
                "sectors": row["sectors"],
                "countries": row["countries"],
                "themes": row["themes"],
                "downloadLink": row["downloadLink"],
                "fileName": row["fileName"],
                "monthYear": row["monthYear"],
                "month": row["month"],
                "year": row["year"],
                "milestoneApprovalDate": row["milestoneApprovalDate"],
                "milestoneEffectivityDate": row["milestoneEffectivityDate"],
                "milestoneSigningDate": row["milestoneSigningDate"],
                "safeguardCategories": row["safeguardCategories"],
                "sourceOfFunding": row["sourceOfFunding"],
                "modalitiesFromWebsite": row["modalitiesFromWebsite"],
                "modalitiesFromDump": row["modalitiesFromDump"],
                "uniqueModalitiesFromDump": row["uniqueModalitiesFromDump"]
            }
        } for index, row in df.iterrows()
    ]
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    helpers.bulk(es, action)
