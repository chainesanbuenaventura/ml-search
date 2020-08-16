import pandas as pd
import subprocess
import datetime
from elasticsearch import Elasticsearch, helpers
from pandasticsearch import Select
from es_pandas import es_pandas
from espandas import Espandas
from argparse import ArgumentParser
import mlflow
from mlflow import log_metric, log_param, log_artifact
import os.path
from os import path
import json

import defaults
import AzureFunctions as af

def deleteIndex(credentials, index):
    """Function to delete an index in elasticsearch
    Returns: None
    Usage:
    >>> import ElasticFunctions as ef
    >>> ef.deleteIndex(credentials, index)
    """
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    es.indices.delete(index=index, ignore=[400, 404])

def getIndex(credentials, indexName):
    """Function to query index data from elasticsearch
    Returns: dataframe with all the index details
    Usage:
    >>> import ElasticFunctions as ef
    >>> df = ef.getLessons(credentials)
    """
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    doc = {
            'size' : 10000,
            'query': {
                'match_all' : {}
        }
    }
    indexDF = pd.DataFrame()
    data = es.search(index=indexName, body=doc, scroll='1m')
    scrollId = data['_scroll_id']
    scrollSize = len(data['hits']['hits'])
    while scrollSize > 0:
        if indexDF.empty:
            indexDF = Select.from_dict(data).to_pandas()
        else:
            indexDF = indexDF.append(Select.from_dict(data).to_pandas())
        data = es.scroll(scroll_id = scrollId, scroll = '1m')
        scrollId = data['_scroll_id']
        scrollSize = len(data['hits']['hits'])
    return indexDF

def getSentences(credentials):
    """Function to query all sentences data from elasticsearch
    Returns: dataframe with all the sentences
    Usage:
    >>> import ElasticFunctions as ef
    >>> df = ef.getLessons(credentials)
    """
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    doc = {
            'size' : 10000,
            'query': {
                'match_all' : {}
        }
    }
    sentencesDF = pd.DataFrame()
    data = es.search(index="sentences", body=doc, scroll='1m')
    scrollId = data['_scroll_id']
    scrollSize = len(data['hits']['hits'])
    while scrollSize > 0:
        if sentencesDF.empty:
            sentencesDF = Select.from_dict(data).to_pandas()
        else:
            sentencesDF = sentencesDF.append(Select.from_dict(data).to_pandas())
        data = es.scroll(scroll_id = scrollId, scroll = '1m')
        scrollId = data['_scroll_id']
        scrollSize = len(data['hits']['hits'])
    return sentencesDF

def getBaseClassification(credentials):
    """Function to query all base classification data from elasticsearch
    Returns: dataframe with all the base classification data
    Usage:
    >>> import ElasticFunctions as ef
    >>> df = ef.getLessons(credentials)
    """
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    doc = {
            'size' : 10000,
            'query': {
                'match_all' : {}
        }
    }
    baseClassificationDF = pd.DataFrame()
    data = es.search(index="base-classification", body=doc, scroll='1m')
    scrollId = data['_scroll_id']
    scrollSize = len(data['hits']['hits'])
    while scrollSize > 0:
        if baseClassificationDF.empty:
            baseClassificationDF = Select.from_dict(data).to_pandas()
        else:
            baseClassificationDF = baseClassificationDF.append(Select.from_dict(data).to_pandas())
        data = es.scroll(scroll_id = scrollId, scroll = '1m')
        scrollId = data['_scroll_id']
        scrollSize = len(data['hits']['hits'])
    return baseClassificationDF

def getLessons(credentials):
    """Description: Function to query all lessons data from elasticsearch
    Returns: dataframe with all the lessons data
    Usage:
    >>> import ElasticFunctions as ef
    >>> df = ef.getLessons(credentials)
    """
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    doc = {
        "query": {
            "term": {
                "isLesson": {
                    "value": True,
                    "boost": 1.0
                }
            }
        }
    }
    lessonsDF = pd.DataFrame()
    data = es.search(index="sentences", body=doc, scroll='1m')
    scrollId = data['_scroll_id']
    scrollSize = len(data['hits']['hits'])
    while scrollSize > 0:
        if lessonsDF.empty:
            lessonsDF = Select.from_dict(data).to_pandas()
        else:
            lessonsDF = lessonsDF.append(Select.from_dict(data).to_pandas())
        data = es.scroll(scroll_id = scrollId, scroll = '1m')
        scrollId = data['_scroll_id']
        scrollSize = len(data['hits']['hits'])
    return lessonsDF

def updateSentences(credentials, updatedDF):
    """Function to update sentences data in elasticsearch
    Returns: None
    Usage:
    >>> import ElasticFunctions as ef
    >>> ef.updateSentences(credentials, updatedDF)
    """
    updatedDF['annotatedBy'] = updatedDF['annotatedBy'].fillna("")
    updatedDF['annotationTitle'] = updatedDF['annotationTitle'].fillna("")
    updatedDF['summary'] = updatedDF['summary'].fillna("")
    updatedDF['context'] = updatedDF['context'].fillna("")
    updatedDF['annotationSummary'] = updatedDF['annotationSummary'].fillna("")
    updatedDF['lastAnnotated'] = updatedDF['lastAnnotated'].fillna("")
    actions = [
        {
            "_index": "sentences",
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
        for i, row  in updatedDF.iterrows()
    ]
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    helpers.bulk(es, actions)

def getMaxReferenceId(credentials):
    """Function to get the maximum reference id from sentences data
    Returns: Max integer referenceId
    """
    sentencesDF = getSentences(credentials)
    referenceIds = [int(r.lstrip('0')) for r in sentencesDF["referenceId"].tolist()]
    return max(referenceIds)

def addNewSentences(credentials, newSentences):
    """Function to add new sentences data to elasticsearch
    Returns: None
    """
    maxReferenceId = getMaxReferenceId(credentials)
    actions = [
        {
            "_index": "sentences",
            "_source": {
                "referenceId": str(maxReferenceId + i + 1).zfill(20),
                "title": row["title"],
                "paragraph": row["paragraph"],
                "themes": row["themes"],
                "sectors": row["sectors"],
                "sectorDiscriminator": defaults.SECTOR_DISCRIMINATOR,
                "countries": row["countries"],
                "lessonStrength": defaults.LESSON_STRENGTH,
                "relatedLessons": defaults.RELATED_LESSONS,
                "topic": defaults.TOPIC,
                "topTopics": defaults.TOP_TOPICS,
                "modalities": row["modalities"],
                "lessonType": defaults.LESSON_TYPE,
                "isLesson": defaults.IS_LESSON, 
                "month": row["month"], 
                "year": row["year"], 
                "annotationTitle": defaults.ANNOTATION_TITLE,
                "summary": defaults.SUMMARY,
                "context":  defaults.CONTEXT,
                "annotatedBy": defaults.ANNOTATED_BY,
                "annotationSummary": defaults.ANNOTATION_SUMMARY,
                "lastAnnotated": defaults.LAST_ANNOTATED,
                "annotationStatus": defaults.ANNOTATION_STATUS,
                "urlToFile": row["urlToFile"],
                "lastUpdated": defaults.LAST_UPDATED,
                "source": defaults.SOURCE
            }
        }
        for i, row in newSentences.iterrows()
    ]
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    helpers.bulk(es, actions)

def getProjectDetails(credentials, projectNumber):
    """Function to get PCR data using projectNumber
    Returns: dataframe with PCR data of a projectNumber
    """
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    doc = {
        "query": {
            "term": {
                "projectNumber": {
                    "value": projectNumber,
                    "boost": 1.0
                }
            }
        }
    }
    data = es.search(index="pcrs", body=doc)
    projectDF = Select.from_dict(data).to_pandas()
    return projectDF

def getTopics(credentials):
    """Function to query all topics data from elasticsearch
    Returns: dataframe with all the topics data
    Usage:
    >>> import ElasticFunctions as ef
    >>> df = ef.getTopics(credentials)
    """
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    doc = {
            'size' : 10000,
            'query': {
                'match_all' : {}
        }
    }
    topicsDF = pd.DataFrame()
    data = es.search(index="topics", body=doc, scroll='1m')
    scrollId = data['_scroll_id']
    scrollSize = len(data['hits']['hits'])
    while scrollSize > 0:
        if topicsDF.empty:
            topicsDF = Select.from_dict(data).to_pandas()
        else:
            topicsDF = topicsDF.append(Select.from_dict(data).to_pandas())
        data = es.scroll(scroll_id = scrollId, scroll = '1m')
        scrollId = data['_scroll_id']
        scrollSize = len(data['hits']['hits'])
    return topicsDF

def saveTopics(credentials, topicsDF):
    """Function to update topics data in elasticsearch
    Returns: None
    """
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    es.indices.delete(index="topics", ignore=[400, 404])
    actions = [
        {
            "_index": "topics",
            "_source": {
                "key": row["key"],
                "keywords": row["keywords"],
                "oldFrequencies": row["oldFrequencies"],
                "numberOfLessons": row["numberOfLessons"],
                "numberOfPCRs": row["oldFrequencies"],
                "frequencies": row["frequencies"],
                "topWord": row["topWord"],
                "x": row["x"],
                "y": row["y"],
                "adjacentTopics": row["adjacentTopics"]
            }
        }
        for i, row  in topicsDF.iterrows()
    ]
    helpers.bulk(es, actions)

def getAnnotatedSentences(credentials):
    """Function to get annotated sentences data from elasticsearch
    Returns: dataframe with all the annotated sentences data
    Usage:
    >>> import ElasticFunctions as ef
    >>> df = ef.getAnnotatedSentences(credentials)
    """
    sentences = getSentences(credentials)
    annotatedDF = sentences.loc[sentences["annotationStatus"].isin(['annotated','forreview'])]
    return annotatedDF

def updateBaseClassification(credentials):
    """Function to update base classifications data with the new annotations
    Returns: None
    Usage:
    >>> import ElasticFunctions as ef
    >>> ef.updateBaseClassification(credentials)
    """
    annotatedDF = getAnnotatedSentences(credentials)
    baseDF = getBaseClassification(credentials)
    existingIds = baseDF["sentencesId"].tolist()
    newAnnotatedDF = annotatedDF.loc[~annotatedDF["_id"].isin(existingIds)]
    newAnnotatedDF = newAnnotatedDF[["_id", "paragraph", "isLesson"]]
    newAnnotatedDF["isLesson"] = newAnnotatedDF["isLesson"].replace(True, 1)
    newAnnotatedDF["isLesson"] = newAnnotatedDF["isLesson"].replace(False, 0)
    newAnnotatedDF = newAnnotatedDF.rename(columns={"_id": "sentencesId"})
    newAnnotatedDF["source"] = "annotation"
    print(newAnnotatedDF)

    actions = [
        {
            "_index": "base-classification",
            "_source": {
                "sentencesId": row["sentencesId"],
                "paragraph": row["paragraph"],
                "isLesson": row["isLesson"],
                "source": row["source"]
            }
        }
        for i, row  in newAnnotatedDF.iterrows()
    ]
    print(actions)
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    helpers.bulk(es, actions)

def updateBaseSummaries(credentials):
    """Function to update base summaries with the new annotations
    Returns: None
    Usage:
    >>> import ElasticFunctions as ef
    >>> ef.updateBaseSummaries(credentials)
    """
    annotatedDF = getAnnotatedSentences(credentials)
    sentencesWithAnntotatedSummaries = annotatedDF[~(annotatedDF["annotationTitle"] == "")]
    sentencesWithAnntotatedSummaries = sentencesWithAnntotatedSummaries.loc[~sentencesWithAnntotatedSummaries["annotationTitle"].isin(defaults.INVALID_ANNOTATION_TITLE)]
    sentencesWithAnntotatedSummaries = sentencesWithAnntotatedSummaries[sentencesWithAnntotatedSummaries["isLesson"] == True]
    sentencesWithAnntotatedSummaries = sentencesWithAnntotatedSummaries[["_id", "paragraph", "annotationTitle"]]
    sentencesWithAnntotatedSummaries = sentencesWithAnntotatedSummaries.rename(columns={"_id": "sentencesId"})
    sentencesWithAnntotatedSummaries["source"] = "annotation"
    # baseDF = getBaseSummaries(credentials)
    # existingIds = baseDF["sentencesId"].tolist()
    # newAnnotatedDF = annotatedDF.loc[~annotatedDF["_id"].isin(existingIds)]

    # newAnnotatedDF["source"] = "annotation"
    actions = [
        {
            "_index": "base-summaries",
            "_source": {
                "sentencesId": row["sentencesId"],
                "paragraph": row["paragraph"],
                "annotationTitle": row["annotationTitle"],
                "source": row["source"]
            }
        }
        for i, row  in sentencesWithAnntotatedSummaries.iterrows()
    ]
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    helpers.bulk(es, actions)

def savePCR(credentials, pcr):
    """Function to save PCR data to elasticsearch
    Returns: None
    Usage:
    >>> import ElasticFunctions as ef
    >>> ef.savePCR(credentials, pcr)
    """
    action = [
        {
            "_index": "pcrs",
            "_source": {
                "projectNumber": pcr["projectNumber"],
                "title": pcr["title"],
                "sectors": pcr["sectors"],
                "countries": pcr["countries"],
                "themes": pcr["themes"],
                "downloadLink": pcr["downloadLink"],
                "fileName": pcr["fileName"],
                "monthYear": pcr["monthYear"],
                "month": pcr["month"],
                "year": pcr["year"],
                "milestoneApprovalDate": "",
                "milestoneEffectivityDate": "",
                "milestoneSigningDate": "",
                "safeguardCategories": "",
                "sourceOfFunding": "",
                # "modalitiesFromWebsite": pcr["countries"],
                "modalitiesFromDump": "",
                "uniqueModalitiesFromDump": "",
                "isExtracted": False,
                "tentative": False
            }
        }
    ]
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    helpers.bulk(es, action)

def updatePCRs(credentials, updatedDF):
    """Function to update pcr data in elasticsearch
    Returns: None
    Usage:
    >>> import ElasticFunctions as ef
    >>> ef.updatePCRs(credentials, updatedDF)
    """
    updatedDF['fileName'] = updatedDF['fileName'].fillna("")
    updatedDF['milestoneApprovalDate'] = updatedDF['milestoneApprovalDate'].fillna("")
    updatedDF['milestoneEffectivityDate'] = updatedDF['milestoneEffectivityDate'].fillna("")
    updatedDF['milestoneSigningDate'] = updatedDF['milestoneSigningDate'].fillna("")
    updatedDF['safeguardCategories'] = updatedDF['safeguardCategories'].fillna("")
    updatedDF['sourceOfFunding'] = updatedDF['sourceOfFunding'].fillna("")
    updatedDF['modalitiesFromWebsite'] = updatedDF['modalitiesFromWebsite'].fillna("")
    updatedDF['modalitiesFromDump'] = updatedDF['modalitiesFromDump'].fillna("")
    updatedDF['uniqueModalitiesFromDump'] = updatedDF['uniqueModalitiesFromDump'].fillna("")

    action = [
        {
            "_index": "pcrs",
            "_id": row["_id"],
            "_source": {
                "projectNumber": row["projectNumber"],
                "isExtracted": row["isExtracted"],
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
        }
        for i, row  in updatedDF.iterrows()
    ]
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    helpers.bulk(es, action)
    
def get_credentials(filename):
    """Load credentials from JSON file
    """
    CREDENTIALS_ROOT = "../credentials"
    with open(os.path.join(CREDENTIALS_ROOT, filename)) as f:
        data = json.load(f)
    return data

def setBlankThemes(credentials):
    PCRsDF = getIndex(credentials, "pcrs")
    for i, r in PCRsDF.iterrows():
        if r["themes"] == "['Others']":
            PCRsDF.at[i, "themes"] = ['Others']
        if len(r["themes"]) == 0:
            PCRsDF.at[i, "themes"] = ['Others']
    updatePCRs(credentials, PCRsDF)

def updateIsExtracted(credentials, PCRDF, value):
    action = [
        {
            "_index": "pcrs",
            "_id": PCRDF["_id"],
            "_source": {
                "projectNumber": PCRDF["projectNumber"],
                "isExtracted": value,
                "tentative": PCRDF["tentative"],
                "title": PCRDF["title"],
                "sectors": PCRDF["sectors"],
                "countries": PCRDF["countries"],
                "themes": PCRDF["themes"],
                "downloadLink": PCRDF["downloadLink"],
                "fileName": PCRDF["fileName"],
                "monthYear": PCRDF["monthYear"],
                "month": PCRDF["month"],
                "year": PCRDF["year"],
                "milestoneApprovalDate": PCRDF["milestoneApprovalDate"],
                "milestoneEffectivityDate": PCRDF["milestoneEffectivityDate"],
                "milestoneSigningDate": PCRDF["milestoneSigningDate"],
                "safeguardCategories": PCRDF["safeguardCategories"],
                "sourceOfFunding": PCRDF["sourceOfFunding"],
                "modalitiesFromWebsite": PCRDF["modalitiesFromWebsite"],
                "modalitiesFromDump": PCRDF["modalitiesFromDump"],
                "uniqueModalitiesFromDump": PCRDF["uniqueModalitiesFromDump"]
            }
        }
    ]
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    helpers.bulk(es, action)

def restoreIndex(credentials, args):
    """Function to restore index data in elasticsearch using an ndjson input file
    """
    deleteIndex(credentials, args.index_name)
    bashCommand = "elasticdump"
    fileName = defaults.DATA_PATH + args.index_name + ".ndjson"
    inputOption = "--input=" + fileName
    outputOption = "--output=http://" + credentials["username"] + ":" + credentials["password"] + "@" + credentials["ip_and_port"] + "/" + args.index_name
    subprocess.run([bashCommand, inputOption, outputOption])

def restoreData(credentials, args):
    """Function to restore index data in elasticsearch using an ndjson file from Azure blob container given a run id
    """
    if af.downloadDataFile(args):
        restoreIndex(credentials, args)
    