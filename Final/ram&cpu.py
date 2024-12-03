import os
import psutil
import time
from ultralytics import YOLO

# Retrieve the model size from the environment variable or use a default
model_size = os.getenv("YOLO_MODEL_SIZE", "11n")  # Default to "11x" if not provided
model_file = f"yolo{model_size}.pt"

# Initialize the YOLO model
model = YOLO(model_file)  # Replace with your YOLO model file
image_folder = r"coco128/images/train2017"  # Replace with your image folder path

# Initialize metrics
cpu_usage_list = []
memory_usage_list = []

# Get the process object for the current Python script
process = psutil.Process(os.getpid())
latency_list = []
num_images_processed = 0

# Monitor and run inference
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    
    # Check if the file is an image
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        print(f"Processing {image_name}...")
        
        # Start the timer
        start_time = time.time()

        # Run inference
        results = model(image_path, save=False, show=False)
        latency_per_img = results[0].speed['preprocess'] + results[0].speed['inference'] + results[0].speed['postprocess']
        latency_list.append(latency_per_img)
        num_images_processed += 1  
        # Record script-specific metrics
        cpu_percent = process.cpu_percent(interval=None)  # CPU usage of the script
        memory_usage = process.memory_percent()  # Memory usage of the script

        # Append metrics to the lists
        cpu_usage_list.append(cpu_percent)
        memory_usage_list.append(memory_usage)

        # Pause briefly to stabilize metric collection
        time.sleep(0.1)

# Calculate averages
average_cpu = sum(cpu_usage_list) / len(cpu_usage_list) if cpu_usage_list else 0
average_memory = sum(memory_usage_list) / len(memory_usage_list) if memory_usage_list else 0
if num_images_processed > 0:
    latency = sum(latency_list) / num_images_processed
else:
    print("No images processed.")

fps = 1000 / latency
# Display results
print("\nInference completed for all images.")
print(f"Average CPU Usage (script-specific): {average_cpu:.2f}%")
print(f"Average Memory Usage (script-specific): {average_memory:.2f}%")
print("Average latency: ", latency)
print("FPS: ",fps)
