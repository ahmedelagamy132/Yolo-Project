import os
import psutil
import time
from ultralytics import YOLO

model_size = os.getenv("YOLO_MODEL_SIZE", "11n")  # Default to "11x" if not provided
model_file = f"yolo{model_size}.pt"

model = YOLO(model_file) 
image_folder = r"coco128/images/train2017"  

cpu_usage_list = []
memory_usage_list = []

process = psutil.Process(os.getpid())
latency_list = []
num_images_processed = 0

for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        print(f"Processing {image_name}...")
        
        start_time = time.time()

        results = model(image_path, save=False, show=False)
        latency_per_img = results[0].speed['preprocess'] + results[0].speed['inference'] + results[0].speed['postprocess']
        latency_list.append(latency_per_img)
        num_images_processed += 1  
        cpu_percent = process.cpu_percent(interval=None)  # CPU usage of the script
        memory_usage = process.memory_percent()  # Memory usage of the script

        cpu_usage_list.append(cpu_percent)
        memory_usage_list.append(memory_usage)

        # Pause briefly to stabilize metric collection
        time.sleep(0.1)

average_cpu = sum(cpu_usage_list) / len(cpu_usage_list) if cpu_usage_list else 0
average_memory = sum(memory_usage_list) / len(memory_usage_list) if memory_usage_list else 0
if num_images_processed > 0:
    latency = sum(latency_list) / num_images_processed
else:
    print("No images processed.")

fps = 1000 / latency
print("\nInference completed for all images.")
print(f"Average CPU Usage (script-specific): {average_cpu:.2f}%")
print(f"Average Memory Usage (script-specific): {average_memory:.2f}%")
print("Average latency: ", latency)
print("FPS: ",fps)
