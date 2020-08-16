from argparse import ArgumentParser
import DocumentExtraction
import sys
sys.path.append("pipeline/functions/DataFunctions")
sys.path.append("../DataFunctions")
from utils import *

if __name__ == "__main__":

    tracking_uri = get_tracking_uri()

# Arguments
    parser = ArgumentParser()
    parser.add_argument("--environment", dest="environment", default="development", required=True, help='Set which environment to run the clustering (development, staging, or production)')
    args = parser.parse_args()

# Scrape website and update PCRs index
    e = DocumentExtraction.Extractor(args, tracking_uri)
    e.extract(args)


