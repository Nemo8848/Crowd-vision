"""
Dataset Preparation Helper
Splits your raw images into train/val sets
and prints the labeling guide
"""

import os
import shutil
import random
from pathlib import Path

# ─── CONFIG ────────────────────────────────────────────────────────────────────

RAW_IMAGES_DIR = "./raw_images"    # Put all your unlabeled images here
DATASET_DIR    = "./dataset"
VAL_SPLIT      = 0.15              # 15% for validation

# ─── SPLIT IMAGES ──────────────────────────────────────────────────────────────

def prepare_dataset():
    raw = Path(RAW_IMAGES_DIR)
    if not raw.exists():
        print(f"❌ Folder not found: {RAW_IMAGES_DIR}")
        print(f"   Create it and put your images inside.")
        return

    images = list(raw.glob("*.jpg")) + list(raw.glob("*.jpeg")) + list(raw.glob("*.png"))
    if not images:
        print("❌ No images found in raw_images/")
        return

    random.shuffle(images)
    val_count   = max(1, int(len(images) * VAL_SPLIT))
    val_images  = images[:val_count]
    train_images= images[val_count:]

    for split, imgs in [("train", train_images), ("val", val_images)]:
        img_dir = Path(DATASET_DIR) / "images" / split
        lbl_dir = Path(DATASET_DIR) / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img in imgs:
            shutil.copy(img, img_dir / img.name)

    print(f"✅ Dataset prepared!")
    print(f"   Total images : {len(images)}")
    print(f"   Train        : {len(train_images)}")
    print(f"   Validation   : {len(val_images)}")
    print(f"\n📁 Structure created at: {DATASET_DIR}/")
    print(f"   images/train/  ← {len(train_images)} images")
    print(f"   images/val/    ← {len(val_images)} images")
    print(f"   labels/train/  ← put .txt label files here after annotation")
    print(f"   labels/val/    ← put .txt label files here after annotation")

# ─── LABELING GUIDE ────────────────────────────────────────────────────────────

LABELING_GUIDE = """
╔══════════════════════════════════════════════════════════════╗
║          HOW TO LABEL YOUR IMAGES (YOLO FORMAT)              ║
╚══════════════════════════════════════════════════════════════╝

OPTION A — Roboflow (Recommended, Free & Easy)
────────────────────────────────────────────
1. Go to  https://roboflow.com  and create a free account
2. Create new project → Object Detection → Class name: "person"
3. Upload all images from  dataset/images/train/  and  val/
4. Annotate: draw bounding boxes around every person
5. Export as "YOLOv8" format
6. Download and place .txt files into:
     dataset/labels/train/
     dataset/labels/val/

OPTION B — LabelImg (Offline, Free)
────────────────────────────────────
1. Install:  pip install labelImg
2. Run:      labelImg
3. Open dir: dataset/images/train/
4. Set save dir: dataset/labels/train/
5. Change format to YOLO (top bar)
6. Draw boxes around every person → label as "person"
7. Save (W key), next image (D key)

YOLO LABEL FORMAT (one .txt per image):
────────────────────────────────────────
Each line = one person:
  0 <cx> <cy> <width> <height>
  
  where all values are 0–1 (normalized by image dimensions)
  class 0 = person

Example:
  0 0.512 0.480 0.120 0.350
  0 0.731 0.510 0.095 0.310

TIPS FOR BETTER ACCURACY:
────────────────────────
✓ Box every visible person, even partially visible ones
✓ Be consistent — if you can see a person, box them
✓ Tight boxes are better than loose ones
✓ Skip if less than ~10px tall (too small to matter)
✓ Aim to annotate ALL images before training
"""

if __name__ == "__main__":
    print(LABELING_GUIDE)
    prepare_dataset()
