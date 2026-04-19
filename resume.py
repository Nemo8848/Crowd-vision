from ultralytics import YOLO

# 1. Load the "last" saved checkpoint
# Find this in: runs/detect/crowd_detector6/weights/last.pt
model = YOLO("/Users/omer/crowd_vision/runs/detect/runs/crowd_detector_medium batch8 NEW/weights/best.pt")

# 2. Resume the training
# YOLO will automatically remember your original epochs, batch size, and data.yaml
model.train(resume=True)