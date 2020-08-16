from argparse import ArgumentParser
import LessonsClustering
import sys
sys.path.append("pipeline/functions/DataFunctions")
sys.path.append("../DataFunctions")
from utils import *

if __name__ == "__main__":

    tracking_uri = get_tracking_uri()

# Arguments
    parser = ArgumentParser()
    parser.add_argument("--mode", dest="mode", default="train", required=True, help='Set mode (fine_tuning, train, predict)')
    parser.add_argument("--environment", dest="environment", default="development", required=True, help='Set which environment to run the clustering (development, staging, or production)')
    parser.add_argument("--run_id_model", dest="run_id_model", default="", required=False, help='Set run id where the model to be used is found (predict mode only)')
    parser.add_argument("--update_related_lessons", dest="update_related_lessons", default="False", required=False, help='Option to update the related lessons (any mode)')
    parser.add_argument("--number_of_topics", dest="number_of_topics", default=0, required=False, type=int, help='Set number of topics (train mode only)')
    parser.add_argument("--alpha", dest="alpha", default=0, required=False, type=float, help='Set alpha (train mode only)')
    parser.add_argument("--beta", dest="beta", default=0, required=False, type=float, help='Set beta (train mode only)')
    parser.add_argument("--max_number_of_topics", dest="max_number_of_topics", default=0, required=False, type=int, help='Set max number of topics (fine_tuning mode only)')
    args = parser.parse_args()

    lc = LessonsClustering.Trainer(tracking_uri, args)
    lc.run(args)

