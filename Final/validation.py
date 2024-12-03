from ultralytics import YOLO

import os 
model_size = os.getenv("YOLO_MODEL_SIZE", "11n")  
model_file = f"yolo{model_size}.pt"

model = YOLO(model_file)  



results = model.val(data="coco128.yaml")
