
# Automated Emotion Data Collection Script
# Author: Same Kiflu
# Description: Collects emotion-labeled images from DuckDuckGo and prepares them for model training.

import os
import random
import hashlib
import requests
import time
from io import BytesIO
from PIL import Image
from ddgs import DDGS

# CONFIGURATION
SEARCH_PHRASES = [
    "happy elderly person face",
    "sad elderly person face",
    "angry elderly person face",
    "surprised elderly person face",
    "neutral elderly person face"
]

MAX_IMAGES = 40  # Number of images per emotion
DATASET_DIR = "emotion_dataset"

RESIZE_IMAGES = True
TARGET_SIZE = (128, 128)
CONVERT_TO_GRAYSCALE = True

SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}
MIN_FILE_SIZE = 50 * 1024  # 50 KB minimum to avoid tiny/broken files


#  UTILITY FUNCTIONS
def ensure_folders():
    """Create dataset split folders if they don't exist"""
    for split in SPLIT_RATIOS.keys():
        os.makedirs(os.path.join(DATASET_DIR, split), exist_ok=True)


def get_md5(content):
    """Generate an MD5 hash to detect duplicates"""
    return hashlib.md5(content).hexdigest()


def save_image(img_data, folder, filename):
    """Save image to a given folder"""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    img_data.save(path)
    return path


# IMAGE DOWNLOAD FUNCTION
def download_images():
    print("Starting image collection...")
    ensure_folders()
    seen_hashes = set()

    with open("download_log.txt", "w") as log, DDGS() as ddgs:
        for phrase in SEARCH_PHRASES:
            print(f"\n🔍 Searching for: '{phrase}'")
            image_paths = []
            count = 0

            # Try fetching images with retry if rate-limited
            try:
                results = ddgs.images(phrase, max_results=MAX_IMAGES)
            except Exception as e:
                print(f"Error getting results for '{phrase}': {e}")
                print("⏳ Waiting 60 seconds before retrying...")
                time.sleep(60)
                try:
                    results = ddgs.images(phrase, max_results=MAX_IMAGES)
                except Exception as e2:
                    print(f"Failed again for '{phrase}': {e2}")
                    continue

            for i, r in enumerate(results):
                img_url = r.get("image")
                if not img_url:
                    continue

                try:
                    response = requests.get(img_url, timeout=10)
                    if len(response.content) < MIN_FILE_SIZE:
                        print("Skipped small image (too small)")
                        continue

                    img_hash = get_md5(response.content)
                    if img_hash in seen_hashes:
                        print("Skipped duplicate image")
                        continue
                    seen_hashes.add(img_hash)

                    img = Image.open(BytesIO(response.content)).convert("RGB")

                    if RESIZE_IMAGES:
                        img = img.resize(TARGET_SIZE)
                    if CONVERT_TO_GRAYSCALE:
                        img = img.convert("L")

                    tmp_folder = os.path.join(DATASET_DIR, "temp")
                    os.makedirs(tmp_folder, exist_ok=True)
                    filename = f"{phrase.replace(' ', '_')}_{i+1}.jpg"
                    tmp_path = save_image(img, tmp_folder, filename)
                    image_paths.append(tmp_path)
                    count += 1
                    print(f"Saved: {tmp_path}")

                except Exception as e:
                    print(f"Skipped one image: {e}")
                    continue

            # Write log summary
            log.write(f"{phrase}: {count} images collected\n")
            print(f"Collected {count} images for '{phrase}'")

            # Split data for training
            split_dataset(image_paths, phrase)

            # Wait between searches to avoid DuckDuckGo rate limits
            print("Cooling down for 20 seconds before next search...")
            time.sleep(20)

    print("\nAll image downloads completed successfully!")


# DATA SPLITTING FUNCTION
def split_dataset(image_paths, phrase):
    """Split downloaded images into train/val/test folders"""
    if not image_paths:
        print(f"No images found for '{phrase}', skipping split.")
        return

    random.shuffle(image_paths)
    total = len(image_paths)
    train_end = int(total * SPLIT_RATIOS["train"])
    val_end = train_end + int(total * SPLIT_RATIOS["val"])

    splits = {
        "train": image_paths[:train_end],
        "val": image_paths[train_end:val_end],
        "test": image_paths[val_end:]
    }

    for split, paths in splits.items():
        split_folder = os.path.join(DATASET_DIR, split, phrase.replace(" ", "_"))
        os.makedirs(split_folder, exist_ok=True)
        for p in paths:
            try:
                filename = os.path.basename(p)
                os.rename(p, os.path.join(split_folder, filename))
            except Exception as e:
                print(f"Error moving {p}: {e}")

    print(f"Split {total} images for '{phrase}' into train/val/test folders.")


# MAIN EXECUTION
if __name__ == "__main__":
    download_images()
