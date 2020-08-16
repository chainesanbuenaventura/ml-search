from argparse import ArgumentParser
import mlflow

import LessonClassification

if __name__ == "__main__":
    credentials = {
        "ip_and_port": "52.163.240.214:9200",
        # "ip_and_port": "127.0.0.1:9200",
        "username": "elastic",
        "password": "Welcometoerni!"
    }

    # Arguments
    parser = ArgumentParser(description="Train a FastAI ULMFit model for Lesson Classification")
    parser.add_argument('--mode', choices=['train', 'predict'], required=True, help='Set mode to train or predict')
    parser.add_argument("--credentials", dest="credentials", type=str, help='JSON file containing Elasticsearch credenta=ials')
    parser.add_argument("--model_file", dest="model_file", type=str, help='Trained model file (predict mode only)')
    parser.add_argument("--deployment_env", choices=['development', 'staging', 'production'], type=str, help='Deployment environment (predict mode only)')
    parser.add_argument("--run_id", dest="run_id", type=str, help='Mlflow run ID (predict mode only)')
    parser.add_argument("--update_sentences", dest="update_sentences", type=str, help='Save predictions to Elasticsearch (predict mode only)')
    args = parser.parse_args()
    
    # Setup Mlflow tracking
#     setup_mlflow()
    
    with mlflow.start_run():
        #run_id = mlflow.active_run().info.run_id
        lc = LessonClassification.Trainer(args)
        if args.mode == "train":
            lc.train(args)
        elif args.mode == "predict":
            lc.predict(args)