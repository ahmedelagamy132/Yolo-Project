# Validates the configured YOLO model against the local COCO128 dataset configuration.
from ultralytics import YOLO

import os
model_size = os.getenv("YOLO_MODEL_SIZE", "11n")
model_file = f"yolo{model_size}.pt"

model = YOLO(model_file)



results = model.val(data="configs/coco128.yaml")
