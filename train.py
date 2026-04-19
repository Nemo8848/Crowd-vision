"""
Crowd Vision — Training Script
High-accuracy configuration for crowd detection & density analysis
Dataset : ~2000 images
Model   : YOLOv8m (medium — best accuracy/speed tradeoff for crowd scenes)
Backend : Apple Silicon MPS (Metal) | CUDA | CPU fallback
Author  : Crowd Vision Project
"""

import yaml
import torch
from pathlib import Path
from ultralytics import YOLO

# ─── CONFIG ────────────────────────────────────────────────────────────────────

DATASET_DIR  = "./dataset"
MODEL_SIZE   = "yolov8m.pt"       # medium: 25M params — handles dense/overlapping people well
EPOCHS       = 100                # generous budget; early stopping will cut it short
IMG_SIZE     = 640               # high-res: critical for small/distant people in crowds
BATCH_SIZE   = 8                  # safe for M4 16GB at 1280px; drop to 4 if MPS OOM
PROJECT      = "./runs"
RUN_NAME     = "crowd_detector_medium batch8 NEW"
MAP50_TARGET = 0.75               # stop training as soon as this mAP50 is reached

# ─── AUGMENTATION ──────────────────────────────────────────────────────────────
# 2000 images is enough to use strong augmentation without overfitting

AUG = dict(
    hsv_h       = 0.015,          # hue shift — handles varying lighting conditions
    hsv_s       = 0.7,            # saturation — indoor/outdoor/overcast variance
    hsv_v       = 0.4,            # brightness — shadows, overexposure, night scenes
    degrees     = 5.0,            # mild rotation — slightly off-level cameras/drones
    translate   = 0.1,            # random crop offset
    scale       = 0.75,           # aggressive zoom — crowd density changes drastically
    shear       = 2.0,            # slight perspective distortion
    perspective = 0.0005,         # simulates overhead/angled camera views (drones, CCTV)
    fliplr      = 0.5,            # horizontal flip — always safe for person detection
    flipud      = 0.0,            # disabled — upside-down people = noise
    mosaic      = 1.0,            # full mosaic on 2000 images — safe, high benefit
    mixup       = 0.0,            # disabled — no benefit for single-class detection
    copy_paste  = 0.3,            # pastes extra persons synthetically — boosts crowd density
    erasing     = 0.4,            # simulates occlusion — people blocking each other
)

# ─── HYPERPARAMETERS ───────────────────────────────────────────────────────────
# 2000 images → can afford slightly higher LR and less regularization than tiny datasets

HYPER = dict(
    optimizer       = "AdamW",    # stable convergence on medium datasets
    lr0             = 0.001,      # initial LR
    lrf             = 0.01,       # final LR = lr0 × lrf (cosine decay)
    momentum        = 0.937,
    weight_decay    = 0.0005,     # L2 regularization
    warmup_epochs   = 3.0,        # shorter warmup — 2000 images doesn't need long warmup
    warmup_bias_lr  = 0.1,
    warmup_momentum = 0.8,
    box             = 7.5,
    cls             = 0.5,        # single class — keep low
    dfl             = 1.5,
    patience        = 30,         # early stopping: halt if no improvement for 30 epochs
    label_smoothing = 0.05,       # reduces overconfidence; safe with 2000 images
)

# ─── DEVICE SETUP ──────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("✅ Apple MPS (Metal) detected — using GPU acceleration")
        return "cpu"
    elif torch.cuda.is_available():
        print(f"✅ CUDA GPU detected: {torch.cuda.get_device_name(0)}")
        return "cuda"
    else:
        print("⚠️  No GPU found — falling back to CPU (will be slow)")
        return "cpu"

# ─── DATASET YAML ──────────────────────────────────────────────────────────────

def create_dataset_yaml() -> str:
    """
    Creates dataset.yaml for YOLO training.

    Expected dataset structure:
        dataset/
          images/
            train/   ← training images (.jpg / .png)
            val/     ← validation images (10–20% of total, ~200-400 from your 2000)
          labels/
            train/   ← YOLO-format .txt files (one per image)
            val/

    YOLO label format per line:
        <class_id> <x_center> <y_center> <width> <height>
        All values normalized to [0, 1]. class_id = 0 (person only)
    """
    config = {
        "path" : str(Path(DATASET_DIR).resolve()),
        "train": "images/train",
        "val"  : "images/val",
        "nc"   : 1,
        "names": ["person"],
    }

    yaml_path = Path(DATASET_DIR) / "dataset.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"📄 Dataset YAML written to: {yaml_path}")
    return str(yaml_path)

# ─── PRE-FLIGHT CHECKS ─────────────────────────────────────────────────────────

def check_dataset(dataset_dir: str) -> None:
    """Validates dataset folder structure and reports image counts."""
    required = ["images/train", "images/val", "labels/train", "labels/val"]
    root     = Path(dataset_dir)
    missing  = [p for p in required if not (root / p).exists()]

    if missing:
        print("\n❌ Missing dataset folders:")
        for m in missing:
            print(f"   • {dataset_dir}/{m}")
        print("\nRun your dataset preparation script first.")
        raise FileNotFoundError("Dataset structure incomplete.")

    train_imgs = list((root / "images/train").iterdir())
    val_imgs   = list((root / "images/val").iterdir())
    print(f"📁 Dataset OK — {len(train_imgs)} train / {len(val_imgs)} val images")

    if len(val_imgs) < 100:
        print("⚠️  Val set seems small — aim for ~200–400 images (10–20% of total)")

# ─── TARGET ACCURACY CALLBACK ──────────────────────────────────────────────────

def stop_at_target_map(trainer) -> None:
    """
    Callback: fires at the end of every epoch.
    Stops training the moment mAP50 hits MAP50_TARGET.
    """
    metrics = trainer.metrics
    if not metrics:
        return

    current_map = metrics.get("metrics/mAP50(B)", 0)
    epoch       = trainer.epoch + 1

    print(f"   📈 Epoch {epoch:>3} — mAP50: {current_map:.4f}  (target: {MAP50_TARGET})")

    if current_map >= MAP50_TARGET:
        print(f"\n🎯 Target mAP50 {MAP50_TARGET} reached at epoch {epoch} — stopping training.")
        trainer.epoch = trainer.epochs  # signals YOLO to end the training loop

# ─── TRAIN ─────────────────────────────────────────────────────────────────────

def train() -> str:
    device    = get_device()
    yaml_path = create_dataset_yaml()

    check_dataset(DATASET_DIR)

    print("\n" + "─" * 60)
    print("🚀  Crowd Vision — Training")
    print("─" * 60)
    print(f"  Model      : {MODEL_SIZE}")
    print(f"  Epochs     : up to {EPOCHS}  (patience={HYPER['patience']})")
    print(f"  mAP50 goal : {MAP50_TARGET}  ← stops here if reached early")
    print(f"  Image size : {IMG_SIZE}px")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  Optimizer  : {HYPER['optimizer']}  lr={HYPER['lr0']}")
    print(f"  Device     : {device}")
    print("─" * 60 + "\n")

    model = YOLO(MODEL_SIZE)
    model.add_callback("on_train_epoch_end", stop_at_target_map)

    results = model.train(
        data    = yaml_path,
        epochs  = EPOCHS,
        imgsz   = IMG_SIZE,
        batch   = BATCH_SIZE,
        device  = device,
        project = PROJECT,
        name    = RUN_NAME,
        save    = True,
        plots   = True,
        verbose = True,
        **AUG,
        **HYPER,
    )

    best_model = Path(PROJECT) / RUN_NAME / "weights" / "best.pt"

    print("\n" + "─" * 60)
    print("✅  Training complete!")
    print(f"📦  Best model : {best_model}")
    print(f"📊  Results    : {Path(PROJECT) / RUN_NAME}")
    print("─" * 60)

    return str(best_model)

# ─── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()