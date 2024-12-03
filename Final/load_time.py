from ultralytics import YOLO
import time
import os
# Measure start time
model_size = os.getenv("YOLO_MODEL_SIZE", "11n")  # Default to "11x" if not provided
model_file = f"yolo{model_size}.pt"

# Initialize the YOLO model
start_time = time.time()

# Load the YOLO model
model = YOLO("yolo11n")  # Replace with your YOLO model file

# Measure end time and calculate load time
end_time = time.time()
load_time = end_time - start_time

print(f"Model load time: {load_time:.3f} seconds")


