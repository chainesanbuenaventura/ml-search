from argparse import ArgumentParser

from utils import *
import ElasticFunctions as ef

if __name__ == "__main__":

    tracking_uri = get_tracking_uri()

# Arguments
    parser = ArgumentParser()
    parser.add_argument("--environment", dest="environment", default="development", required=True, help='Set which environment to run the clustering (development, staging, or production)')
    parser.add_argument("--mode", dest="mode", required=True, help='Set mode to perform (restore_data)')
    parser.add_argument("--module", dest="module", required=True, help='Set module where data will be accessed (documentExtraction, paragraphExtraction, lessonsClustering, lessonsClassification, lessonsSummarization')
    parser.add_argument("--run_id", dest="run_id", required=True, help='Set run id where the data will be accessed')
    parser.add_argument("--index_name", dest="index_name", required=True, help='Set index name (pcrs, sentences, topics, base-classification)')
    args = parser.parse_args()


    credentials = ef.get_credentials("localcredentials.json")
    if args.environment == "staging":
        credentials = ef.get_credentials("stagingcredentials.json")
    elif args.environment == "production":
        credentials = ef.get_credentials("prodcredentials.json")

    if args.mode == "restore_data":
        ef.restoreData(credentials, args)