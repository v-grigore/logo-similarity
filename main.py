# Suppress TensorFlow warnings
import logging

from src.constants import LOGS_CONFIG_FILE
from src.logo_extractor import extract_logos
from src.feature_extractor import extract_features
from src.logo_clustering import cluster_logos_h

def main():
    # Set up logging
    logging.config.fileConfig(LOGS_CONFIG_FILE)

    extract_logos()

    extract_features()

    cluster_logos_h()

if __name__ == "__main__":
    main()