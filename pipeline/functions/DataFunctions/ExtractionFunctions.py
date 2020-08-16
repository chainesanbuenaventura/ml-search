import os

import ElasticFunctions as ef

def markExtractedPCRs(credentials):
    """Function to mark all PCRs that are already extracted by checking existing file names
    """
    PCRsDF = ef.getIndex(credentials, "pcrs")
    fileNames = PCRsDF["fileName"].tolist()
    isExtracted = []
    extractedPCRsFileNames = []
    for path, subdirs, files in os.walk("data/docx/"):
        for name in files:
            docxFileName = os.path.join(path, name)
            if not docxFileName[-5:] == ".docx":
                continue
            extractedPCRsFileNames.append(name[:name.find("_Page")])
    extractedPCRsFileNames = list(set(extractedPCRsFileNames))
    for f in fileNames:
        if f in extractedPCRsFileNames:
            isExtracted.append(True)
        else:
            isExtracted.append(False)


def markExtractedPCRs2(credentials):
    """Function to mark all PCRs that are already extracted by checking existing project numbers
    """
    sentencesDF = ef.getIndex(credentials, "sentences")
    PCRsDF = ef.getIndex(credentials, "pcrs")
    foundProjectNumbers = []
    isExtracted = []
    projectNumbers = PCRsDF["projectNumber"].tolist()
    for i, r in sentencesDF.iterrows():
        if r["projectNumber"] in projectNumbers:
            foundProjectNumbers.append(r["projectNumber"])
    foundProjectNumbers = list(set(foundProjectNumbers))
    for p in projectNumbers:
        if p in foundProjectNumbers:
            isExtracted.append(True)
        else:
            isExtracted.append(False)
    PCRsDF["isExtracted"] = isExtracted
    ef.updatePCRs(credentials, PCRsDF)


def markExtractedPCRs3(credentials):
    """Function to mark all PCRs that are already extracted by checking existing download links
    """
    sentencesDF = ef.getIndex(credentials, "sentences")
    PCRsDF = ef.getIndex(credentials, "pcrs")
    isExtracted = []
    urls = list(set(sentencesDF["urlToFile"].tolist()))
    for i, r in PCRsDF.iterrows():
        if r["downloadLink"] in urls:
            isExtracted.append(True)
        else:
            isExtracted.append(False)
    PCRsDF["isExtracted"] = isExtracted
    ef.updatePCRs(credentials, PCRsDF)


def getPCRsToExtract(credentials):
    PCRsDF = ef.getIndex(credentials, "pcrs")
    extractedPCRsFileNames = []
    for path, subdirs, files in os.walk("data/docx/"):
        for name in files:
            docxFileName = os.path.join(path, name)
            if not docxFileName[-5:] == ".docx":
                continue
            extractedPCRsFileNames.append(name[:name.find("_Page")])
    extractedPCRsFileNames = list(set(extractedPCRsFileNames))
    hasDocx = []
    for f in PCRsDF["fileName"].tolist():
        if f in extractedPCRsFileNames:
            hasDocx.append(True)
        else:
            hasDocx.append(False)
    PCRsDF["hasDocx"] = hasDocx

    sentencesDF = ef.getIndex(credentials, "sentences")
    fileNames = sentencesDF["projectNumber"].tolist()
    inSentences = []
    tentative = []
    for i, r in PCRsDF.iterrows():
        if r["isExtracted"] == False and r["hasDocx"] == True:
            tentative.append(True)
        else:
            tentative.append(False)
        if r["projectNumber"] in fileNames:
            inSentences.append(True)
        else:
            inSentences.append(False)
    PCRsDF["inSentences"] = inSentences
    PCRsDF["tentative"] = tentative
    ef.updatePCRs(credentials, PCRsDF)
