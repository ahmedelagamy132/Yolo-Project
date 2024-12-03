from ultralytics import YOLO

import os 
model_size = os.getenv("YOLO_MODEL_SIZE", "11n")  # Default to "11x" if not provided
model_file = f"yolo{model_size}.pt"

# Initialize the YOLO model
model = YOLO(model_file)  # Replace with your YOLO model file

# Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run inference with the YOLO11n model on the 'bus.jpg' image
results = model.val(data="coco128.yaml")