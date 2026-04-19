"""
Crowd Vision — Inference Script
Real-time crowd detection & risk prediction
Supports: webcam | video file | single image
"""

import cv2
import sys
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from dataclasses import dataclass
from typing import Literal

# ─── CONFIG ────────────────────────────────────────────────────────────────────

MODEL_PATH  = "/Users/omer/crowd_vision/runs/detect/runs/crowd_detector_medium batch8 NEW/weights/best.pt"
CONF_THRESH = 0.35        # lower = catches more people, higher = fewer false positives
IOU_THRESH  = 0.45        # NMS overlap threshold

# ─── RISK THRESHOLDS ───────────────────────────────────────────────────────────
# Two signals combined: raw head count + area density (% of frame covered by boxes)
# Tune COUNT_* for your camera's field of view (wider FOV = raise the numbers)

COUNT_LOW      = 10        # 0–5 people       → LOW
COUNT_MEDIUM   = 25       # 6–15 people      → MEDIUM
COUNT_HIGH     = 50       # 16–30 people     → HIGH
                          # 31+              → CRITICAL

# Secondary density check: total box area / frame area
# Catches dense crowds even when heads overlap (YOLO under-counts)
DENSITY_MEDIUM   = 0.25   # 25% of frame covered → at least MEDIUM
DENSITY_HIGH     = 0.50   # 50% of frame covered → at least HIGH
DENSITY_CRITICAL = 0.70   # 70% of frame covered → CRITICAL

# ─── RISK ENGINE ───────────────────────────────────────────────────────────────

@dataclass
class CrowdAnalysis:
    person_count  : int
    density_pct   : float           # % of frame area covered by bounding boxes
    risk_level    : Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    risk_color    : tuple           # BGR for OpenCV
    recommendation: str

RISK_CONFIG = {
    "LOW": {
        "color": (0, 200, 80),      # green
        "rec"  : "✅  Normal conditions. No action needed.",
    },
    "MEDIUM": {
        "color": (0, 165, 255),     # orange
        "rec"  : "👁  Monitor closely. Consider managing crowd flow.",
    },
    "HIGH": {
        "color": (0, 60, 255),      # red
        "rec"  : "⚠️  Alert staff. Limit entry. Open exit routes.",
    },
    "CRITICAL": {
        "color": (0, 0, 200),       # dark red
        "rec"  : "🚨 EVACUATE or stop entry immediately!",
    },
}

def calculate_risk(
    person_count: int,
    boxes,
    frame_w: int,
    frame_h: int,
) -> CrowdAnalysis:

    # ── Density: total box area as % of frame ──────────────────────────────────
    frame_area  = frame_w * frame_h
    box_area    = 0.0
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        box_area += (x2 - x1) * (y2 - y1)
    density_pct = box_area / frame_area   # 0.0 – 1.0

    # ── Risk from count ────────────────────────────────────────────────────────
    if person_count <= COUNT_LOW:
        risk = "LOW"
    elif person_count <= COUNT_MEDIUM:
        risk = "MEDIUM"
    elif person_count <= COUNT_HIGH:
        risk = "HIGH"
    else:
        risk = "CRITICAL"

    # ── Upgrade risk from density (catches overlapping/occluded crowds) ────────
    if density_pct >= DENSITY_CRITICAL:
        risk = "CRITICAL"
    elif density_pct >= DENSITY_HIGH and risk in ("LOW", "MEDIUM"):
        risk = "HIGH"
    elif density_pct >= DENSITY_MEDIUM and risk == "LOW":
        risk = "MEDIUM"

    cfg = RISK_CONFIG[risk]
    return CrowdAnalysis(
        person_count   = person_count,
        density_pct    = round(density_pct * 100, 1),   # store as % for display
        risk_level     = risk,
        risk_color     = cfg["color"],
        recommendation = cfg["rec"],
    )

# ─── DRAW OVERLAY ──────────────────────────────────────────────────────────────

# Risk level → short display label
RISK_LABELS = {
    "LOW"     : "LOW",
    "MEDIUM"  : "MED",
    "HIGH"    : "HIGH",
    "CRITICAL": "CRIT",
}

def draw_overlay(frame: np.ndarray, boxes, analysis: CrowdAnalysis) -> np.ndarray:
    h, w = frame.shape[:2]
    color = analysis.risk_color

    # ── Bounding boxes ─────────────────────────────────────────────────────────
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, f"{conf:.0%}",
            (x1, max(y1 - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA,
        )

    # ── Semi-transparent top bar ───────────────────────────────────────────────
    bar_h   = 110
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # ── Risk badge ─────────────────────────────────────────────────────────────
    badge_w = 145
    cv2.rectangle(frame, (8, 6), (badge_w, 103), color, -1)
    cv2.rectangle(frame, (8, 6), (badge_w, 103), (255, 255, 255), 1)  # border

    label = RISK_LABELS[analysis.risk_level]
    font_scale = 1.3 if len(label) <= 3 else 1.0
    cv2.putText(
        frame, label,
        (18, 68),
        cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA,
    )

    # ── Stats + recommendation (all in top bar) ────────────────────────────────
    cv2.putText(
        frame, f"People : {analysis.person_count}",
        (badge_w + 15, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.80, (230, 230, 230), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"Density: {analysis.density_pct:.1f}% of frame",
        (badge_w + 15, 68),
        cv2.FONT_HERSHEY_SIMPLEX, 0.68, (175, 175, 175), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, analysis.recommendation,
        (badge_w + 15, 100),
        cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA,
    )

    return frame

# ─── WEBCAM / VIDEO INFERENCE ──────────────────────────────────────────────────

def run_inference(source: str | int = 0) -> None:
    device = "mps" if (
        torch.backends.mps.is_available() and torch.backends.mps.is_built()
    ) else "cuda" if torch.cuda.is_available() else "cpu"

    print(f"🔍 Loading model : {MODEL_PATH}")
    print(f"📡 Device        : {device}")

    if not Path(MODEL_PATH).exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        print("   Update MODEL_PATH at the top of this file.")
        return

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Cannot open source: {source}")
        return

    print("▶  Running — press Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        results = model(
            frame,
            conf    = CONF_THRESH,
            iou     = IOU_THRESH,
            device  = device,
            verbose = False,
        )

        boxes    = results[0].boxes
        analysis = calculate_risk(len(boxes), boxes, w, h)
        frame    = draw_overlay(frame, boxes, analysis)

        print(
            f"\r👥 {analysis.person_count:>4} people | "
            f"Density: {analysis.density_pct:>5.1f}% | "
            f"Risk: {analysis.risk_level:<8}",
            end="", flush=True,
        )

        cv2.imshow("Crowd Vision  —  Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Done.")

# ─── SINGLE IMAGE INFERENCE ────────────────────────────────────────────────────

def analyze_image(image_path: str, save_output: bool = True) -> None:
    device = "mps" if (
        torch.backends.mps.is_available() and torch.backends.mps.is_built()
    ) else "cuda" if torch.cuda.is_available() else "cpu"

    if not Path(MODEL_PATH).exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH)
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"❌ Image not found: {image_path}")
        return

    h, w     = frame.shape[:2]
    results  = model(frame, conf=CONF_THRESH, iou=IOU_THRESH, device=device, verbose=False)
    boxes    = results[0].boxes
    analysis = calculate_risk(len(boxes), boxes, w, h)
    frame    = draw_overlay(frame, boxes, analysis)

    print(f"\n📊 Results for: {image_path}")
    print(f"   People detected : {analysis.person_count}")
    print(f"   Frame density   : {analysis.density_pct:.1f}%")
    print(f"   Risk level      : {analysis.risk_level}")
    print(f"   Recommendation  : {analysis.recommendation}")

    if save_output:
        out_path = Path(image_path).stem + "_analyzed.jpg"
        cv2.imwrite(out_path, frame)
        print(f"   Saved to        : {out_path}")

    cv2.imshow("Crowd Vision — Result", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ─── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python infer.py webcam      → live webcam")
        print("  python infer.py video.mp4   → video file")
        print("  python infer.py image.jpg   → single image")
        sys.exit(0)

    arg = sys.argv[1].lower()

    if arg == "webcam":
        run_inference(0)
    elif arg.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        analyze_image(sys.argv[1])
    else:
        run_inference(sys.argv[1])    # treat as video path