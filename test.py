from ultralytics import YOLO
from pathlib import Path

model = YOLO("/Users/omer/crowd_vision/runs/detect/runs/crowd_detector_medium batch8 NEW/weights/best.pt")

results = model.predict(
    source  = "/Users/omer/crowd_vision/dataset/images/test",  # your test folder
    imgsz   = 640,
    conf    = 0.35,      # detection confidence threshold
    save    = True,      # saves annotated images
    project = "./runs",
    name    = "test_results",
)

# print count per image
for r in results:
    count = len(r.boxes)
    print(f"{Path(r.path).name}: {count} people detected")