import os

# Directories
LOGOS_DIR = os.path.join(os.getcwd(), "logos")
LOGS_CONFIG_FILE = os.path.join(os.getcwd(), "config/logging.conf")

# URL for Clearbit API
CLEARBIT_LOGO_API_URL = "https://logo.clearbit.com/"

# Default timeout for HTTP requests
DEFAULT_TIMEOUT = 5

# Default dataset path
DEFAULT_DATASET = os.path.join(os.getcwd(), "data/logos.snappy.parquet")

# Headers for HTTP requests
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Loggers
LOGGER_INFO = "info"
LOGGER_ERROR = "error"

# HTTP status codes
HTTP_OK = 200
HTTP_NOT_FOUND = 404
