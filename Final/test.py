from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Customize validation settings
validation_results = model.val(data=r"coco128.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6, device="cpu")