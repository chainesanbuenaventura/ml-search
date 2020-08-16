# Global
DATA_PATH = "../../../data/"
MODEL_PATH = "../../../models/"
CREDENTIALS_PATH = "../credentials/"
CREDENTIALS_PATH_2 = "../../credentials/"
CREDENTIALS_PATH_3 = "../../credentials/"

# sentences index
SECTOR_DISCRIMINATOR = 0
LESSON_STRENGTH = -1
RELATED_LESSONS = []
TOPIC = -1
TOP_TOPICS = []
LESSON_TYPE = "Self Evaluation"
IS_LESSON = False
ANNOTATION_TITLE = ""
SUMMARY = ""
CONTEXT = ""
ANNOTATED_BY = ""
ANNOTATION_SUMMARY = ""
LAST_ANNOTATED = ""
ANNOTATION_STATUS = "draft"
LAST_UPDATED = ""
SOURCE = "tagged"

# Azure
MLFLOW_CONTAINER = "mlflow-container"
PCR_CONTAINER = "pcrs"
DOCX_CONTAINER = "pcrs-docx"

# anotation
INVALID_ANNOTATION_TITLE = ["Not a lesson", 
    "Not a Lesson", 
    "Not a lesson", 
    "(no lesson)", 
    "not a lesson", 
    "Not a lesson", 
    "Not a lesson This is a finding", 
    "Not a lesson (but a finding)", 
    "(no lessons)", 
    "(not a lesson)"
]