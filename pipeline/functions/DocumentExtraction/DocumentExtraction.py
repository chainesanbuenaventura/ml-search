import pandas as pd
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import sys
import os
import mlflow
from mlflow import log_metric, log_param, log_artifact

sys.path.append("pipeline/functions/DataFunctions")
sys.path.append("../DataFunctions")
import defaults
import ElasticFunctions as ef
import AzureFunctions as af
import MLFlowFunctions as mf


class Extractor():
    def __init__(self, args, tracking_uri):
        
        # Website details
        self.domain = 'https://www.adb.org/'
        self.headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
        self.nPages = 65
        self.startPage = 0

        # Credentials
        credentials = ef.get_credentials("localcredentials.json")
        if args.environment == "staging":
            credentials = ef.get_credentials("stagingcredentials.json")
        elif args.environment == "production":
            credentials = ef.get_credentials("prodcredentials.json")
        self.credentials = credentials
        self.tracking_uri = tracking_uri

    def extract(self, args):

    # mlflow logs
        experiment_name = "dev-DocumentExtraction"
        if args.environment == "production":
            experiment_name = "DocumentExtraction"
        elif args.environment == "staging":
            experiment_name = "staging-DocumentExtraction"
        mlflow.set_experiment(experiment_name)
        client = mlflow.tracking.MlflowClient()

        with mlflow.start_run():
            log_param("environment", args.environment)
            
    # Current PCRs
            PCRsDF = ef.getIndex(self.credentials, "pcrs")
            currentProjectNumbers = PCRsDF["projectNumber"].tolist()
            currentDownloadLinks = PCRsDF["downloadLink"].tolist()

            count = 0
            new = 0
            noSectors = pd.DataFrame()
            projectNumbers = []
            links = []

            for page in range(self.startPage, self.nPages):
                print("Page: " + str(page + 1))
                url = self.domain + 'projects/documents/doctype/Project%252FProgram%20Completion%20Reports?page=' + str(page)
                r = requests.get(url, headers=self.headers)
                html = r.text
                soup = BeautifulSoup(html, 'html.parser')
                itemTitles = soup.findAll("div", {"class": "item-title"})
                for title in itemTitles:
                    pcr = {}
                    count = count + 1

    # Download Link
                    documentLink = self.domain + title.find("a").get('href')
                    if documentLink in currentDownloadLinks:
                        print(": Already scraped: " + documentLink)
                        continue
                        
                    documentLink = self.domain + title.find("a").get('href')
                    r2 = requests.get(documentLink, headers=self.headers)
                    html2 = r2.text
                    soup2 = BeautifulSoup(html2, 'html.parser')
                    asideRegion = soup2.find("aside", {"class": "region"})
                    downloadLink = ''
                    if(asideRegion):
                        downloadLink = asideRegion.find("a").get('href')

    # Project Number
                    articleTags = soup2.find("div", {"class": "article-tags"})
                    fieldItems = articleTags.findAll("ul", {"class": "field-items"})
                    projectNumberField = fieldItems[0].find("li", {"class": "field-item"})
                    projectNumber = projectNumberField.find("a").text
                    
                    if projectNumber in currentProjectNumbers:
                        print("Already scraped: " + documentLink)
                        continue
                    
                    new = new + 1

    # Title
                    titleText = title.find("a").text
                    print("Count: " + str(count) + ", Current Document: " + titleText)
                    
                    pcr["title"] = titleText
                    pcr["projectNumber"] = projectNumber
                    pcr["downloadLink"] = downloadLink

    # Month Year
                    dateDisplay = soup2.find("span", {"class": "date-display-single"})
                    pcr["monthYear"] = dateDisplay.text
                    pcr["month"] = dateDisplay.text.split(" ")[0]
                    pcr["year"] = dateDisplay.text.split(" ")[1]
                    
    # Sectors
                    sectorFound = True
                    sectors = []
                    if(len(fieldItems)>2):
                        sectorFields = fieldItems[2].findAll("li", {"class": "field-item"})
                        for sectorField in sectorFields:
                            sectors.append(sectorField.find("a").text)
                    else:
                        sectorFound = False

                    projectNumberLink = projectNumberField.find("a").get('href')
                    r3 = requests.get(self.domain + projectNumberLink + '#project-pds')
                    html3 = r3.text
                    soup3 = BeautifulSoup(html3, 'html.parser')
                    themes = []
                    if(soup3.find("table", {"class": "pds"})):
                        pdsTableTrs = soup3.find("table", {"class": "pds"}).findAll("tr")
                        for tr in pdsTableTrs:
                            trtds = tr.findAll("td")
                            if(trtds):

    # Themes
                                if(trtds[0].text == 'Strategic Agendas'):
                                    if(len(trtds)>1):
                                        themesRaw = str(trtds[1]).split(' <br/>')
                                        themesRaw[0] = themesRaw[0][4:]
                                        themes = themesRaw[:-1]
                    
    # Sectors (2nd chance)
                                if not sectorFound:
                                    if(trtds[0].text == 'Sector / Subsector'):
                                        if(len(trtds)>1):
                                            if(trtds[1].find("p")):
                                                sector = trtds[1].find("p").text.split("/")[0].rstrip()
                                                sectors.append(sector)
                                            else:
                                                print(documentLink)
                    if sectors == []:
                        projectNumbers.append(projectNumber)
                        links.append(downloadLink)
                    
                    pcr["sectors"] = sectors
                    pcr["themes"] = themes

    # Country
                    countries = []
                    if(len(fieldItems)>1):
                        countryFields = fieldItems[1].findAll("li", {"class": "field-item"})
                        for countryField in countryFields:
                            countries.append(countryField.find("a").text)
                    pcr["countries"] = countries
                    
    # File
                    startFileName = 0
                    for i in range(len(downloadLink)-1, 0, -1):
                        if downloadLink[i] == '/':
                            startFileName = i + 1
                            break

                    fileName = downloadLink[startFileName:]
                    folderName = defaults.DATA_PATH + "PCRs/"
                    pcr["fileName"] = fileName
                    if(downloadLink and downloadLink[0] != '/'):
                        r4 = requests.get(downloadLink)
                        with open(folderName + fileName, 'wb') as f:
                            f.write(r4.content)
                        af.savePCRPDF(folderName + fileName, fileName)
                        os.remove(folderName + fileName)
                    else:
                        with open(folderName + fileName + '-invalid.txt', 'w') as f:
                            f.write('Invalid file')
                    ef.savePCR(self.credentials, pcr)

            noSectors["projectNumber"] = projectNumbers
            noSectors["downloadLink"] = links
            noSectors.to_excel(defaults.DATA_PATH + "no-sectors.xlsx")
    
    # Set blank themes to "Others"
            ef.setBlankThemes(self.credentials)
        

    # Backup pcrs index
            mf.backupIndex(self.credentials, "pcrs")

    def loadInitial(self):
        """Function to load the initial PCR data (before pipeline development) using an input excel file
        """
        PCRsDF = pd.read_excel(defaults.DATA_PATH + "pcrs-20200415.xlsx")
        ef.saveIndex(self.credentials, PCRsDF, "pcrs")
