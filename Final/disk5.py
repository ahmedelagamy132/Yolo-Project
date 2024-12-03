import psutil
import time
import os
from PIL import Image
from ultralytics import YOLO

# Retrieve the model size from the environment variable or use a default
model_size = os.getenv("YOLO_MODEL_SIZE", "11n")  # Default to "11x" if not provided
model_file = f"yolo{model_size}.pt"

# Initialize the YOLO model
model = YOLO(model_file)  # Repla

# Path to the directory containing test images
image_directory = r"coco128/images/train2017"

# Collect all image file paths from the directory
image_paths = [
    os.path.join(image_directory, img) 
    for img in os.listdir(image_directory) 
    if img.lower().endswith(('.png', '.jpg', '.jpeg'))
]

# Pre-load all images into memory
images = {path: Image.open(path) for path in image_paths}

# Initialize accumulators for metrics
total_elapsed_time = 0
total_read_bytes = 0
total_write_bytes = 0
total_images = len(image_paths)

if total_images == 0:
    print("No images found in the directory.")
else:
    # Create psutil Process object for current process
    process = psutil.Process(os.getpid())

    # Process each image
    for image_path, image in images.items():
        # Monitor per-process I/O before model inference
        io_start = process.io_counters()

        # Measure inference time
        start_time = time.time()
        results = model(image, save=True, show=False)
        end_time = time.time()

        # Monitor per-process I/O after model inference
        io_end = process.io_counters()

        # Calculate I/O statistics for the current image
        elapsed_time = end_time - start_time
        read_bytes = (io_end.read_bytes - io_start.read_bytes) / (1024**2)  # MB
        write_bytes = (io_end.write_bytes - io_start.write_bytes) / (1024**2)  # MB

        # Accumulate metrics
        total_elapsed_time += elapsed_time
        total_read_bytes += read_bytes
        total_write_bytes += write_bytes

        print(f"Processed {os.path.basename(image_path)}: {elapsed_time:.2f}s, {read_bytes:.2f}MB read, {write_bytes:.2f}MB written")

    # Compute averages
    avg_elapsed_time = total_elapsed_time / total_images
    avg_read_bytes = total_read_bytes / total_images
    avg_write_bytes = total_write_bytes / total_images

    # Calculate average I/O speeds
    avg_read_speed = avg_read_bytes / avg_elapsed_time if avg_elapsed_time > 0 else 0
    avg_write_speed = avg_write_bytes / avg_elapsed_time if avg_elapsed_time > 0 else 0

    # Display summary
    print("\nSummary for all images:")
    print(f"Total Images: {total_images}")
    print(f"Average Elapsed Time: {avg_elapsed_time:.4f} seconds")
    print(f"Average Read Bytes: {avg_read_bytes:.4f} MB")
    print(f"Average Write Bytes: {avg_write_bytes:.4f} MB")
    print(f"Average Disk Read Speed: {avg_read_speed:.4f} MB/s")
    print(f"Average Disk Write Speed: {avg_write_speed:.4f} MB/s")
