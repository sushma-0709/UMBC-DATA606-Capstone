#!/usr/bin/env python3
import requests
from pathlib import Path
from datetime import datetime, timedelta

# Litecoin data types
LITECOIN_DATA_TYPES = ["blocks", "transactions", "inputs", "outputs", "addresses"]

# Base URL
BASE_URL = "https://gz.blockchair.com"

# Save location
DESTINATION = Path("/root/UMBC-DATA606-Capstone/data/litecoin")


def download_file(url: str, output_path: Path):
    """Download a file from URL and save it."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading: {url}")
        response = requests.get(url, stream=True, timeout=60)
        if response.status_code == 404:
            print(f"File not found: {url}")
            return False
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Saved: {output_path}")
        return True
    except Exception as e:
        print(f"Failed: {url} ({e})")
        return False


def download_litecoin(num_days: int = 1):
    """Download Litecoin data dumps for the past num_days."""
    today = datetime.now()
    for d in range(1, num_days + 1):
        date = (today - timedelta(days=d)).strftime("%Y%m%d")
        print(f"\n=== Litecoin data for {date} ===")
        for dtype in LITECOIN_DATA_TYPES:
            if dtype == "addresses":
                filename= f"blockchair_litecoin_{dtype}_latest.tsv.gz"
            else:
                filename = f"blockchair_litecoin_{dtype}_{date}.tsv.gz"
            url = f"{BASE_URL}/litecoin/{dtype}/{filename}"
            output_path = DESTINATION / dtype / filename
            download_file(url, output_path)


if __name__ == "__main__":
    download_litecoin(num_days=2)  
    print("\nDone")
