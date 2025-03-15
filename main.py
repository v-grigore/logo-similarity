import logging

from src.constants import LOGS_CONFIG_FILE
from src.logo_extractor import extract_logos

def main():
    # Set up logging
    logging.config.fileConfig(LOGS_CONFIG_FILE)

    extract_logos()

if __name__ == "__main__":
    main()