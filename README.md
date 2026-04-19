# 🧠 Crowd Vision — Crowd Detection & Risk Prediction
### Optimized for Apple Silicon (M4/MPS) · YOLOv8

---

## 📁 Project Structure

```
crowd_vision/
├── raw_images/          ← Put ALL your original images here
├── dataset/             ← Auto-generated after prepare_dataset.py
│   ├── images/train/
│   ├── images/val/
│   ├── labels/train/    ← Your .txt annotation files go here
│   └── labels/val/
├── runs/                ← Training output (auto-created)
│   └── crowd_detector/
│       └── weights/
│           └── best.pt  ← Your final trained model
├── prepare_dataset.py   ← Step 1
├── train.py             ← Step 3
├── infer.py             ← Step 4
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup (One Time)

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Step-by-Step Workflow

### STEP 1 — Prepare Your Dataset
```bash
# Put all your images into the raw_images/ folder, then:
python prepare_dataset.py
```
This will split images into train/val sets and print the labeling guide.

---

### STEP 2 — Label Your Images
Use **Roboflow** (easiest) or **LabelImg** (offline).

Run `python prepare_dataset.py` to see the full guide printed to terminal.

**Goal:** Every image needs a matching `.txt` file with bounding boxes around people.

---

### STEP 3 — Train the Model
```bash
python train.py
```
- Takes ~30–90 min on M4 depending on dataset size
- Best model auto-saved to `runs/crowd_detector/weights/best.pt`
- Watch for `mAP50` metric — aim for > 0.7 (70%)

---

### STEP 4 — Run Inference

```bash
# Live webcam
python infer.py webcam

# Single image
python infer.py path/to/image.jpg

# Video file
python infer.py /Users/omer/Downloads/archivee/Crowd-UIT/Video/japan.mp4
```

---

## 📊 Risk Levels

| Level    | Meaning                         | Action                        |
|----------|---------------------------------|-------------------------------|
| 🟢 LOW    | Normal crowd density            | No action needed              |
| 🟠 MEDIUM | Crowd building up               | Monitor, manage flow          |
| 🔴 HIGH   | Dangerously dense               | Alert staff, limit entry      |
| ⛔ CRITICAL| Extreme density                | Stop entry, initiate evacuation|

---

## 🎯 Tips for Better Results

- **More labeled data = better model.** Label all images carefully.
- If accuracy is low, increase `EPOCHS` in `train.py` (try 150–200).
- Tune `DENSITY_LOW` and `DENSITY_MEDIUM` in `infer.py` based on your camera's field of view.
- For a wide-angle camera covering a large area, lower the thresholds.
- For a close-up camera, raise them.

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| `mps` not found | Update macOS to 13+ and PyTorch to 2.1+ |
| Out of memory | Reduce `BATCH_SIZE` to 4 in `train.py` |
| Low accuracy | Add more labeled images, increase epochs |
| Slow training | Normal on CPU. MPS should be 5–10x faster |
