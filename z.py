import os
import tarfile
import random
import shutil
from PIL import Image
from tqdm import tqdm

IMAGENET_DIR = "/dpl/datasets/imagenet-1k/train"
OUTPUT_DIR = "/dpl/datasets/imagenet256/train"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1) Extract to a temporary directory
TEMP_DIR = "/dpl/datasets/imagenet256_tmp"
os.makedirs(TEMP_DIR, exist_ok=True)

tars = [f for f in os.listdir(IMAGENET_DIR) if f.endswith(".tar")]
all_images = []

print("Extracting .tar files (no overwrite to original)...")
for tarf in tqdm(tars):
    tar_path = os.path.join(IMAGENET_DIR, tarf)
    with tarfile.open(tar_path) as t:
        t.extractall(TEMP_DIR)

print("Scanning extracted images...")
for root, _, files in os.walk(TEMP_DIR):
    for name in files:
        img_path = os.path.join(root, name)
        try:
            with Image.open(img_path) as img:
                if img.width >= 256 and img.height >= 256:
                    all_images.append(img_path)
        except:
            continue  # skip broken files

print(f"Found >=256px images: {len(all_images)}")

# 2) Sample 300k
random.shuffle(all_images)
selected_images = all_images[:300000]

print("Copying selected images...")
for idx, src in tqdm(enumerate(selected_images)):
    dst = os.path.join(OUTPUT_DIR, f"{idx:07d}.JPEG")
    shutil.copy2(src, dst)  # ← copy to keep original safe

print("Done! 🎯")
