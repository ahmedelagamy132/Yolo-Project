# Purpose: run the supplementary CPU-only YOLO validation variant.
# Input: `configs/coco128.yaml` and fixed YOLO11n model weights.
# Output: Ultralytics validation metrics calculated on the CPU.
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Customize validation settings
validation_results = model.val(data=r"configs/coco128.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6, device="cpu")
