import mlflow

mlflow.set_tracking_uri("http://40.112.217.252:5000/")

try:
    # Production
    mlflow.create_experiment("DocumentExtraction", artifact_location="wasbs://mlflow-container@mlpipelines.blob.core.windows.net/production/documentExtraction")
    mlflow.create_experiment("ParagraphExtraction", artifact_location="wasbs://mlflow-container@mlpipelines.blob.core.windows.net/production/paragraphExtraction")
    mlflow.create_experiment("LessonsClassification", artifact_location="wasbs://mlflow-container@mlpipelines.blob.core.windows.net/production/lessonsClassification")
    mlflow.create_experiment("LessonsClustering", artifact_location="wasbs://mlflow-container@mlpipelines.blob.core.windows.net/production/lessonsClustering")
    mlflow.create_experiment("LessonsSummarization", artifact_location="wasbs://mlflow-container@mlpipelines.blob.core.windows.net/production/lessonsSummarization")

    # Staging
    mlflow.create_experiment("staging-DocumentExtraction", artifact_location="wasbs://mlflow-container@mlpipelines.blob.core.windows.net/staging/documentExtraction")
    mlflow.create_experiment("staging-ParagraphExtraction", artifact_location="wasbs://mlflow-container@mlpipelines.blob.core.windows.net/staging/paragraphExtraction")
    mlflow.create_experiment("staging-LessonsClassification", artifact_location="wasbs://mlflow-container@mlpipelines.blob.core.windows.net/staging/lessonsClassification")
    mlflow.create_experiment("staging-LessonsClustering", artifact_location="wasbs://mlflow-container@mlpipelines.blob.core.windows.net/staging/lessonsClustering")
    mlflow.create_experiment("staging-LessonsSummarization", artifact_location="wasbs://mlflow-container@mlpipelines.blob.core.windows.net/staging/lessonsSummarization")

    # Develop
    mlflow.create_experiment("dev-DocumentExtraction", artifact_location="wasbs://mlflow-container@mlpipelines.blob.core.windows.net/development/documentExtraction")
    mlflow.create_experiment("dev-ParagraphExtraction", artifact_location="wasbs://mlflow-container@mlpipelines.blob.core.windows.net/development/paragraphExtraction")
    mlflow.create_experiment("dev-LessonsClassification", artifact_location="wasbs://mlflow-container@mlpipelines.blob.core.windows.net/development/lessonsClassification")
    mlflow.create_experiment("dev-LessonsClustering", artifact_location="wasbs://mlflow-container@mlpipelines.blob.core.windows.net/development/lessonsClustering")
    mlflow.create_experiment("dev-LessonsSummarization", artifact_location="wasbs://mlflow-container@mlpipelines.blob.core.windows.net/development/lessonsSummarization")
except:
    print("Experiments are already created. No need to re-create them.")