#!/usr/bin/env python3
"""
Prepare dataset for super-resolution training.
Downloads DIV2K dataset, splits into train/val/test, and generates LR_x2 images.
"""

import os
import sys
import shutil
import urllib.request
import zipfile
from pathlib import Path
from PIL import Image

# Dataset configuration
DATA_ROOT = Path("data")
RAW_DIR = DATA_ROOT / "raw" / "images"
TRAIN_HR = DATA_ROOT / "train" / "HR"
TRAIN_LR = DATA_ROOT / "train" / "LR_x2"
VAL_HR = DATA_ROOT / "val" / "HR"
VAL_LR = DATA_ROOT / "val" / "LR_x2"
TEST_HR = DATA_ROOT / "test" / "HR"
TEST_LR = DATA_ROOT / "test" / "LR_x2"

# DIV2K URLs
DIV2K_TRAIN_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
DIV2K_VALID_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"

# Split configuration
TRAIN_COUNT = 600
VAL_COUNT = 50
TEST_COUNT = 50


def download_file(url, dest):
    """Download file with progress."""
    if dest.exists():
        print(f"  Already exists: {dest}")
        return
    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest}")


def extract_zip(zip_path, extract_to):
    """Extract zip file."""
    print(f"  Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    print(f"  Extracted to {extract_to}")


def create_dirs():
    """Create all necessary directories."""
    for d in [RAW_DIR, TRAIN_HR, TRAIN_LR, VAL_HR, VAL_LR, TEST_HR, TEST_LR]:
        d.mkdir(parents=True, exist_ok=True)


def generate_lr(hr_path, lr_path, scale=2):
    """Generate LR image from HR using Bicubic downsampling."""
    img = Image.open(hr_path).convert("RGB")
    w, h = img.size
    lr_img = img.resize((w // scale, h // scale), Image.BICUBIC)
    lr_img.save(lr_path)


def copy_and_generate_lr(src_hr_dir, dst_hr_dir, dst_lr_dir, image_names):
    """Copy HR images and generate corresponding LR images."""
    for name in image_names:
        src = src_hr_dir / name
        if not src.exists():
            continue
        dst_hr = dst_hr_dir / name
        dst_lr = dst_lr_dir / name
        shutil.copy2(src, dst_hr)
        generate_lr(src, dst_lr)


def main():
    create_dirs()

    # Download DIV2K
    print("=" * 50)
    print("Step 1: Download DIV2K dataset")
    print("=" * 50)

    train_zip = RAW_DIR / "DIV2K_train_HR.zip"
    valid_zip = RAW_DIR / "DIV2K_valid_HR.zip"

    download_file(DIV2K_TRAIN_URL, train_zip)
    download_file(DIV2K_VALID_URL, valid_zip)

    # Extract
    print("\n" + "=" * 50)
    print("Step 2: Extract archives")
    print("=" * 50)

    train_extract = RAW_DIR / "DIV2K_train_HR"
    valid_extract = RAW_DIR / "DIV2K_valid_HR"

    if not train_extract.exists():
        extract_zip(train_zip, RAW_DIR)
    if not valid_extract.exists():
        extract_zip(valid_zip, RAW_DIR)

    # Get all image paths
    print("\n" + "=" * 50)
    print("Step 3: Organize images")
    print("=" * 50)

    train_images = sorted([f.name for f in train_extract.glob("*.png")])
    valid_images = sorted([f.name for f in valid_extract.glob("*.png")])

    print(f"  Train HR images: {len(train_images)}")
    print(f"  Valid HR images: {len(valid_images)}")

    # Split
    train_split = train_images[:TRAIN_COUNT]
    val_split = valid_images[:VAL_COUNT]
    test_split = valid_images[VAL_COUNT : VAL_COUNT + TEST_COUNT]

    print(f"\n  Split:")
    print(f"    Train: {len(train_split)}")
    print(f"    Val:   {len(val_split)}")
    print(f"    Test:  {len(test_split)}")

    # Copy and generate LR
    print("\n" + "=" * 50)
    print("Step 4: Generate LR_x2 images")
    print("=" * 50)

    copy_and_generate_lr(train_extract, TRAIN_HR, TRAIN_LR, train_split)
    print(
        f"  Train: {len(list(TRAIN_HR.glob('*.png')))} HR, {len(list(TRAIN_LR.glob('*.png')))} LR"
    )

    copy_and_generate_lr(valid_extract, VAL_HR, VAL_LR, val_split)
    print(
        f"  Val:   {len(list(VAL_HR.glob('*.png')))} HR, {len(list(VAL_LR.glob('*.png')))} LR"
    )

    copy_and_generate_lr(valid_extract, TEST_HR, TEST_LR, test_split)
    print(
        f"  Test:  {len(list(TEST_HR.glob('*.png')))} HR, {len(list(TEST_LR.glob('*.png')))} LR"
    )

    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)


if __name__ == "__main__":
    main()
