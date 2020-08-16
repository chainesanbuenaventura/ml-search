from argparse import ArgumentParser
import mlflow

import LessonSummarization

if __name__ == "__main__":
    credentials = {
        "ip_and_port": "52.163.240.214:9200",
        # "ip_and_port": "127.0.0.1:9200",
        "username": "elastic",
        "password": "Welcometoerni!"
    }

    # Arguments
    parser = ArgumentParser(description="Train a Torch Presumm model for Lesson Summarization")
    parser.add_argument('--mode', choices=['train', 'predict'], required=True, help='Set mode to train or predict')
    parser.add_argument("--credentials", dest="credentials", type=str, help='JSON file containing Elasticsearch credentsials')
    parser.add_argument("--deployment_env", choices=['development', 'staging', 'production'], type=str, help='Deployment environment (predict mode only)')
    parser.add_argument("--run_id", dest="run_id", type=str, help='Mlflow run ID (predict mode only)')
    parser.add_argument("--update_sentences", dest="update_sentences", type=bool, help='Save summaries to Elasticsearch (predict mode only)')
    args = parser.parse_args()
    
    # Setup Mlflow tracking
#     setup_mlflow()
    
    with mlflow.start_run():
#         run_id = mlflow.active_run().info.run_id
        ls = LessonSummarization.Trainer(args)
        ls.run(args)