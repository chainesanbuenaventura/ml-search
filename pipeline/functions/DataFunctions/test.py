from TestFunctions import *
import ElasticFunctions as ef
import AzureFunctions as af

if True:
    """Contains functions for data fixes and tests
    """

    # af.saveFile("../lesson-summarization/models/", "bertextabs.pt", "files")
    # print(get_tracking_uri())

    # credentials = ef.get_credentials("localcredentials.json")
    # prodCredentials = ef.get_credentials("prodcredentials.json")
    # stagingCredentials = ef.get_credentials("stagingcredentials.json")

    # projectDF = ef.getProjectDetails(credentials, "37220-013")
    # producePCRs(credentials)
    # downloadIndex(prodCredentials, "sentences")
    # ef.updateBaseClassification(credentials)
    # ef.setBlankThemes(credentials)
    # recomputeFileNames(credentials)
    # renameDocx()
    # saveAllPCRDOCX()
    # countCurrentPCRs(credentials)

    # pcrsDF = ef.getIndex(credentials, "pcrs")
    # pcrsDF.to_excel("data/index.xlsx")

    # mf.backupIndex(credentials, "topics")

    # setIsExtracted("35174-082-pcr-en.pdf")

    # ef.getMaxReferenceId(credentials)

    # PCRDF = ef.getProjectDetails(credentials, "35174-082")
    # ef.updateIsExtracted(credentials, PCRDF.iloc[0], False)

    # mf.backupIndex(credentials, "sentences")

    # prodSentencesDF = ef.getIndex(prodCredentials, "sentences")
    # saveIndex(stagingCredentials,  prodSentencesDF, "temp-sentences")
    # resetRelatedLessons(stagingCredentials, "temp-sentences")
    # PERFORM REINDEX
    # PERFORM ADD ALIAS
    # PERFORM updateRelatedLessons
    # resetColumns(stagingCredentials, "temp-sentences")
    # PERFORM REINDEX


    # saveBaseClassifications(prodCredentials)


    # MIGRATION
    # indexName = "sentences"
    # df = ef.getIndex(stagingCredentials, indexName)
    # saveIndex(credentials, df, indexName)

    # recoverProjectNumbers(prodCredentials)
    # updatePCRIds(prodCredentials)

    # saveBaseClassifications(stagingCredentials)
    # ef.updateBaseClassification(stagingCredentials)
    # df = ef.getBaseClassification(stagingCredentials)
    # df.to_excel("data/base.xlsx")
    # downloadIndex(stagingCredentials, "base-classification")
    # exf.markExtractedPCRs3(credentials)

    # FIX EMPTY string relatedLessons
    # resetRelatedLessons(prodCredentials, "sentences")

    # saveBaseSummaries(stagingCredentials)
    # ef.updateBaseSummaries(stagingCredentials)

    # downloadIndex(stagingCredentials, "base-summaries")

    # df = ef.getBaseClassification(stagingCredentials)
    # df = ef.getBaseSummaries(stagingCredentials)
    # df.to_excel("base-summaries.xlsx")

