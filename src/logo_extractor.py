from enum import Enum
import logging.config
import os
import warnings
from bs4 import BeautifulSoup
import cairosvg
import pandas as pd
import requests
from urllib.parse import urljoin
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import logging

from src.constants import *

class Result(Enum):
    CLEARBIT = 1
    FAVICON = 2
    SCRAPED = 3
    BROKEN = 4
    FAILED = 5

def setup(parquet_file):
    # Create the 'logos' directory if it doesn't exist
    os.makedirs(LOGOS_DIR, exist_ok=True)

    # Load the Parquet file
    df = pd.read_parquet(parquet_file, engine="pyarrow")

    # Drop duplicates in the 'domain' column
    df = df.drop_duplicates(subset="domain")

    return df

def get_logo_from_html(domain_name):
    # Set up loggers
    error_logger = logging.getLogger("error")

    try:
        response = requests.get(f"https://{domain_name}", timeout=DEFAULT_TIMEOUT, headers=HEADERS, allow_redirects=True)
        if response.status_code != HTTP_OK:
            error_logger.error(f"Failed to download HTML for {domain_name}: {response.status_code}")
            return Result.BROKEN

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Search for the <link> tags with rel="apple-touch-icon", rel="icon", rel="shortcut icon" and rel="logo"
        link_tags = soup.find_all("link", rel=["apple-touch-icon", "icon", "shortcut icon", "logo"])

        for link in link_tags:
            href = link.get('href')
            if href:
                if not href.startswith('http'):
                    # If the href is relative, make it absolute
                    href = urljoin(f"https://{domain_name}", href)
                
                # Download and save the logo
                logo_response = requests.get(href, timeout=DEFAULT_TIMEOUT, headers=HEADERS, allow_redirects=True)
                if logo_response.status_code == HTTP_OK:
                    if href.endswith(".svg"):
                        # Convert SVG to PNG
                        png_data = cairosvg.svg2png(bytestring=logo_response.content)
                        with open(os.path.join(LOGOS_DIR, f"{domain_name}.png"), 'wb') as f:
                            f.write(png_data)
                        return Result.SCRAPED

                    img = Image.open(BytesIO(logo_response.content))
                    img.save(os.path.join(LOGOS_DIR, f"{domain_name}.png"))
                    return Result.SCRAPED

    except requests.exceptions.RequestException as e:
        error_logger.error(f"Request error for {domain_name}: {e}")
        return Result.BROKEN
    except Exception as e:
        error_logger.error(f"Error saving logo from {domain_name}: {e}")
        return Result.FAILED

def save_logo(domain_name):
    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Set up loggers
    error_logger = logging.getLogger("error")

    # Base URL for Clearbit API
    base_url = CLEARBIT_LOGO_API_URL

    try:
        # Download the logo from Clearbit
        response = requests.get(base_url + domain_name, timeout=DEFAULT_TIMEOUT, headers=HEADERS)

        if response.status_code == HTTP_OK:
            img = Image.open(BytesIO(response.content))
            img.save(os.path.join(LOGOS_DIR, f"{domain_name}.png"))
            return Result.CLEARBIT
        
        if response.status_code != HTTP_NOT_FOUND:
            error_logger.error(f"Failed to download logo for {domain_name}: {response.status_code}")
            error_logger.error(f"Response content: {response.content}")
            return Result.FAILED

        # Try favicon.ico for 404 responses
        response = requests.get(f"https://{domain_name}/favicon.ico", timeout=DEFAULT_TIMEOUT, headers=HEADERS)

        if response.status_code == HTTP_OK:
            if not response.content:
                return get_logo_from_html(domain_name)

            img = Image.open(BytesIO(response.content))
            img.save(os.path.join(LOGOS_DIR, f"{domain_name}.png"))
            return Result.FAVICON

        if response.status_code != HTTP_NOT_FOUND:
            error_logger.error(f"Failed to download favicon for {domain_name}: {response.status_code}")
            error_logger.error(f"Response content: {response.content}")
            return Result.FAILED

        # If all else fails, scrape the logo from the website
        return get_logo_from_html(domain_name)

    except Exception as e:
        return get_logo_from_html(domain_name)

def extract_logos(parquet_file=DEFAULT_DATASET):
    info_logger = logging.getLogger(LOGGER_INFO)
    
    # Logo counters
    clearbit_count = 0
    favicon_count = 0
    scraped_logo_count = 0
    broken_website_count = 0
    
    df = setup(parquet_file)

    # Using ThreadPoolExecutor to process domains concurrently
    tp = ThreadPoolExecutor(max_workers=10)
    results = list(tqdm(tp.map(save_logo, df["domain"]), total=len(df["domain"])))
    tp.shutdown()

    for result in results:
        if result == Result.CLEARBIT:
            clearbit_count += 1
        elif result == Result.FAVICON:
            favicon_count += 1
        elif result == Result.SCRAPED:
            scraped_logo_count += 1
        elif result == Result.BROKEN:
            broken_website_count += 1

    print("Logo extraction completed! ✅")

    info_logger.info(f"Clearbit logos saved: {clearbit_count}")
    info_logger.info(f"Favicon logos saved: {favicon_count}")
    info_logger.info(f"Scraped logos saved: {scraped_logo_count}")
    info_logger.info(f"Total logos saved: {clearbit_count + favicon_count + scraped_logo_count} / {len(df['domain'])}")
    info_logger.info(f"Success rate: {(clearbit_count + favicon_count + scraped_logo_count) / len(df['domain']) * 100:.2f}%\n")

    info_logger.info(f"Working websites: {len(df['domain']) - broken_website_count}")
    info_logger.info(f"Broken websites: {broken_website_count}")
    info_logger.info(f"Succes rate for working websites: {(clearbit_count + favicon_count + scraped_logo_count) / (len(df['domain']) - broken_website_count) * 100:.2f}%")
