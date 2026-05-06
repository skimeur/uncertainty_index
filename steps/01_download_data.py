"""
01_download_data.py -- Download ECB-SPF individual forecasts.

Downloads the ZIP archive of individual probability distributions from the
ECB Survey of Professional Forecasters and extracts CSV files to
data/SPF_individual_forecasts/.

Author: Eric Vansteenberghe
Reference:
    Vansteenberghe, E. (2026). "Uncertain and Asymmetric Forecasts."
    Banque de France Working Paper.
"""
import sys
import os
import shutil
import tempfile
import zipfile
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SPF_ZIP_URL, DATA_DIR, RAW_DIR


def download_spf(force=False):
    """Download and unzip SPF individual forecasts."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "SPF_individual_forecasts.zip"

    # Check if already downloaded
    existing = list(RAW_DIR.glob("*.csv"))
    if existing and not force:
        print(f"Already have {len(existing)} CSV files in {RAW_DIR}. Use force=True to re-download.")
        return

    print(f"Downloading SPF data from {SPF_ZIP_URL} ...")
    resp = requests.get(SPF_ZIP_URL, timeout=120)
    resp.raise_for_status()

    with open(zip_path, "wb") as f:
        f.write(resp.content)
    print(f"Downloaded {len(resp.content) / 1024:.0f} KB → {zip_path}")

    print("Extracting ...")
    with tempfile.TemporaryDirectory(dir=DATA_DIR) as tmp_dir:
        tmp_raw_dir = DATA_DIR / os.path.basename(tmp_dir) / RAW_DIR.name
        tmp_raw_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_raw_dir)

        csv_files = list(tmp_raw_dir.rglob("*.csv"))
        if not csv_files:
            raise RuntimeError("Downloaded archive did not contain any CSV files.")

        if RAW_DIR.exists():
            shutil.rmtree(RAW_DIR)
        shutil.move(str(tmp_raw_dir), str(RAW_DIR))

    csv_count = len(list(RAW_DIR.glob("*.csv")))
    print(f"Extracted {csv_count} CSV files to {RAW_DIR}")

    # Clean up zip
    zip_path.unlink()
    print("Done.")


if __name__ == "__main__":
    force = "--force" in sys.argv
    download_spf(force=force)
