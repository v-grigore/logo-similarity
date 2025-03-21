import argparse
import logging
import os

from src.constants import *

def input_file(file_path):
    if file_path is None:
        print("Using default dataset.")
        return DEFAULT_DATASET

    if os.path.isfile(file_path) and file_path.endswith('.parquet'):
        return file_path
    else:
        print("Invalid file. Using default dataset.")
        return DEFAULT_DATASET

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str, nargs="?", 
                      help="path to input parquet file")
    parser.add_argument("-s", "--save", action="store_true",
                      help="save processed output to clustered_logos/")
    
    args = parser.parse_args()
    
    # Use default if no path provided
    input_path = input_file(args.file_path)
    saving_clusters = args.save

    from src.logo_extractor import extract_logos
    from src.feature_extractor import extract_features
    from src.logo_clustering import cluster_logos_h

    # Set up logging
    logging.config.fileConfig(LOGS_CONFIG_FILE)

    extract_logos(input_path)

    extract_features()

    cluster_logos_h(saving_clusters)

if __name__ == "__main__":
    main()