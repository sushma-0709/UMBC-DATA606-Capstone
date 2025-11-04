#!/usr/bin/env python3
"""
Litecoin blockchain data downloader.

This module downloads Litecoin blockchain data from Blockchair's public API
and saves it to local storage for further processing.

Usage:
    python Scripts/download_data.py

Environment Variables:
    DATA_DIR: Directory to save downloaded data (default: ./data/litecoin)
    NUM_DAYS_TO_DOWNLOAD: Number of days to download (default: 2)
"""

import logging
import logging.handlers
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
from requests.exceptions import ConnectionError, RequestException, Timeout

# Configure logging
def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Configure application-wide logging."""
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_path / "download_data.log",
        maxBytes=10_000_000,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()

# Litecoin data types
LITECOIN_DATA_TYPES = ["blocks", "transactions", "inputs", "outputs", "addresses"]

# Base URL
BASE_URL = "https://gz.blockchair.com"

# Save location - configurable via environment variable
DESTINATION = Path(os.getenv("DATA_DIR", "./data/litecoin"))
NUM_DAYS = int(os.getenv("NUM_DAYS_TO_DOWNLOAD", "2"))


def download_file(
    url: str, output_path: Path, max_retries: int = 3
) -> bool:
    """
    Download a file from URL and save it with retry logic.

    Args:
        url: URL to download from
        output_path: Path where to save the file
        max_retries: Maximum number of retry attempts

    Returns:
        True if download successful, False otherwise
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading (attempt {attempt + 1}/{max_retries}): {url}")

            response = requests.get(url, stream=True, timeout=60)

            if response.status_code == 404:
                logger.warning(f"File not found (404): {url}")
                return False

            if response.status_code == 403:
                logger.warning(f"Access forbidden (403): {url}")
                return False

            response.raise_for_status()

            # Validate file size
            file_size = 0
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        file_size += len(chunk)

            if file_size < 1000:  # Less than 1KB is suspicious
                logger.warning(f"Downloaded file suspiciously small ({file_size} bytes): {output_path}")
                output_path.unlink()  # Delete the small file
                return False

            logger.info(f"Successfully saved: {output_path} ({file_size} bytes)")
            return True

        except (ConnectionError, Timeout) as e:
            logger.warning(
                f"Network error on attempt {attempt + 1}/{max_retries}: {e}"
            )
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            return False

        except RequestException as e:
            logger.error(f"HTTP request failed: {url} - {e}")
            return False

        except IOError as e:
            logger.error(f"File I/O error: {output_path} - {e}")
            return False

        except Exception as e:
            logger.error(f"Unexpected error downloading {url}: {e}", exc_info=True)
            return False

    return False


def download_litecoin(num_days: int = NUM_DAYS) -> None:
    """
    Download Litecoin data dumps for the past num_days.

    Args:
        num_days: Number of days of data to download
    """
    logger.info(f"Starting Litecoin data download for {num_days} day(s)")

    today = datetime.now()
    total_files = 0
    successful_downloads = 0
    failed_downloads = 0

    for d in range(1, num_days + 1):
        date = (today - timedelta(days=d)).strftime("%Y%m%d")
        logger.info(f"Processing Litecoin data for {date}")

        for dtype in LITECOIN_DATA_TYPES:
            total_files += 1

            if dtype == "addresses":
                filename = f"blockchair_litecoin_{dtype}_latest.tsv.gz"
            else:
                filename = f"blockchair_litecoin_{dtype}_{date}.tsv.gz"

            url = f"{BASE_URL}/litecoin/{dtype}/{filename}"
            output_path = DESTINATION / dtype / filename

            if download_file(url, output_path):
                successful_downloads += 1
            else:
                failed_downloads += 1

    # Summary
    logger.info(
        f"Download complete: {successful_downloads}/{total_files} files downloaded successfully"
    )
    if failed_downloads > 0:
        logger.warning(f"{failed_downloads} files failed to download")


if __name__ == "__main__":
    try:
        download_litecoin()
        logger.info("Download process finished successfully")
    except KeyboardInterrupt:
        logger.info("Download process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error during download: {e}", exc_info=True)
        sys.exit(1)
