# =============================================================================
# 🎓 DATA DOWNLOAD SCRIPT — Getting the Training Data from Kaggle
# =============================================================================
#
# WHY DO WE NEED THIS?
# ---------------------
# The training CSV (6 million rows, ~500MB) is too large for GitHub.
# We .gitignore it so it never gets committed. But then how does anyone
# get the data to train the model?
#
# Answer: Kaggle's API. Kaggle is a platform that hosts ML datasets.
# This script uses their API to download the data programmatically —
# no manual clicking, no sharing large files.
#
# WHAT IS THE DATASET?
# ----------------------
# PaySim — a synthetic financial transaction simulator.
# - 6.3 million transactions
# - Simulates mobile money transfers in Africa
# - Only TRANSFER and CASH_OUT transactions can be fraud
# - Fraud rate: ~0.13% (very imbalanced — that's why we use class weights)
# Source: https://www.kaggle.com/datasets/ealaxi/paysim1
#
# HOW TO USE THIS SCRIPT:
# ------------------------
# Step 1: Get your Kaggle API key
#   → Go to kaggle.com → click your profile → Account → API → "Create New Token"
#   → This downloads kaggle.json to your Downloads folder
#
# Step 2: Place kaggle.json in the right location
#   → Mac/Linux: ~/.kaggle/kaggle.json
#   → Windows:   C:\Users\<username>\.kaggle\kaggle.json
#   → Or: set environment variables KAGGLE_USERNAME and KAGGLE_KEY
#
# Step 3: Run this script
#   → python download_data.py
#
# =============================================================================

import os
import sys
import zipfile
from pathlib import Path


def check_kaggle_credentials() -> bool:
    """
    Check that Kaggle credentials exist before trying to download.

    🎓 WHERE KAGGLE LOOKS FOR CREDENTIALS (in order):
    1. Environment variables: KAGGLE_USERNAME and KAGGLE_KEY
    2. Config file: ~/.kaggle/kaggle.json

    Returns True if credentials found, False otherwise.
    """
    # Check environment variables first
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        print("✓ Kaggle credentials found in environment variables")
        return True

    # Check config file
    kaggle_config = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_config.exists():
        print(f"✓ Kaggle credentials found at {kaggle_config}")
        return True

    # Neither found — print helpful instructions
    print("✗ Kaggle credentials not found!")
    print()
    print("To fix this:")
    print("  1. Go to https://www.kaggle.com → Account → API → 'Create New Token'")
    print("  2. This downloads kaggle.json")
    print("  3. Move it to ~/.kaggle/kaggle.json")
    print()
    print("  Mac/Linux:")
    print("    mkdir -p ~/.kaggle")
    print("    mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json")
    print("    chmod 600 ~/.kaggle/kaggle.json  # secure the file")
    print()
    print("  OR set environment variables:")
    print("    export KAGGLE_USERNAME=your_username")
    print("    export KAGGLE_KEY=your_api_key")
    return False


def install_kaggle_if_needed() -> bool:
    """
    Check if the kaggle package is installed. If not, prompt to install.

    🎓 WHY CHECK THIS?
    The kaggle package is a CLI + Python library for Kaggle's API.
    It's not in requirements.txt because it's only needed for data download,
    not for running the model or API.
    """
    try:
        import kaggle  # noqa: F401
        return True
    except ImportError:
        print("The 'kaggle' package is not installed.")
        print("Install it with:  pip install kaggle")
        return False


def download_paysim(data_dir: Path) -> Path:
    """
    Download the PaySim dataset from Kaggle.

    🎓 WHAT HAPPENS STEP BY STEP:
    1. kaggle.api.dataset_download_files() sends an HTTP request to Kaggle
    2. Kaggle authenticates your credentials
    3. The dataset is downloaded as a .zip file
    4. We extract the CSV from the zip
    5. We delete the zip (no need to keep it)

    Args:
        data_dir: Directory to save the CSV into

    Returns:
        Path to the downloaded CSV file
    """
    import kaggle

    data_dir.mkdir(parents=True, exist_ok=True)

    dataset = "ealaxi/paysim1"
    zip_path = data_dir / "paysim1.zip"
    csv_path = data_dir / "PS_20174392719_1491204439457_log.csv"
    final_path = data_dir / "transactions.csv"

    # Skip if already downloaded
    if final_path.exists():
        size_mb = final_path.stat().st_size / (1024 * 1024)
        print(f"✓ Dataset already exists at {final_path} ({size_mb:.0f} MB)")
        print("  Delete it and re-run to force re-download.")
        return final_path

    print(f"Downloading PaySim dataset from Kaggle...")
    print(f"Dataset: {dataset}")
    print(f"Saving to: {data_dir}")
    print("(This is ~500MB — may take a few minutes on slower connections)\n")

    # Download
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        dataset,
        path=str(data_dir),
        unzip=False,  # We'll handle extraction manually
    )

    # Find the downloaded zip
    zip_files = list(data_dir.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"Download completed but no zip found in {data_dir}")
    zip_path = zip_files[0]

    print(f"\nExtracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Show what's inside
        names = zf.namelist()
        print(f"  Contents: {names}")
        zf.extractall(data_dir)

    # Clean up zip
    zip_path.unlink()
    print(f"✓ Zip deleted (no longer needed)")

    # Rename to a simpler name if original name exists
    if csv_path.exists():
        csv_path.rename(final_path)
        print(f"✓ Renamed to: {final_path.name}")
    else:
        # Find whatever CSV was extracted
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            csv_files[0].rename(final_path)
            print(f"✓ Renamed to: {final_path.name}")
        else:
            raise FileNotFoundError("Extraction succeeded but no CSV found")

    size_mb = final_path.stat().st_size / (1024 * 1024)
    print(f"\n✓ Dataset ready: {final_path} ({size_mb:.0f} MB)")
    return final_path


def main():
    """
    Main entry point — runs all checks then downloads.
    """
    print("=" * 55)
    print("  PaySim Fraud Detection Dataset Downloader")
    print("=" * 55)
    print()

    # Step 1: Check kaggle package
    if not install_kaggle_if_needed():
        sys.exit(1)

    # Step 2: Check credentials
    if not check_kaggle_credentials():
        sys.exit(1)

    # Step 3: Download
    project_root = Path(__file__).parent
    data_dir = project_root / "data"

    try:
        csv_path = download_paysim(data_dir)
        print()
        print("=" * 55)
        print("  All done! Next steps:")
        print("=" * 55)
        print(f"  1. Train the model:")
        print(f"     python main.py --mode train")
        print(f"")
        print(f"  2. Quick test with 10% of data (faster):")
        print(f"     python main.py --mode train --sample-frac 0.1")
        print(f"")
        print(f"  3. Start the API after training:")
        print(f"     python api.py")

    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("\nTry manually:")
        print("  1. Go to https://www.kaggle.com/datasets/ealaxi/paysim1")
        print("  2. Click Download")
        print("  3. Extract and rename CSV to: data/transactions.csv")
        sys.exit(1)


if __name__ == "__main__":
    main()
