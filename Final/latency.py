import os
from ultralytics import YOLO

model = YOLO("yolo11s.pt")

image_folder = r"coco128/images/train2017"
latency_list = []
num_images_processed = 0  # Keep track of the number of images processed

for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        print(f"Processing {image_name}...")
        
        results = model(image_path, save=False, show=False)
        # print(results[0].speed)
        latency_per_img = results[0].speed['preprocess'] + results[0].speed['inference'] + results[0].speed['postprocess']
        latency_list.append(latency_per_img)
        num_images_processed += 1  # Increment for each image processed

# Calculate the average latency for the images processed
if num_images_processed > 0:
    latency = sum(latency_list) / num_images_processed
    print("Average latency:", latency)
else:
    print("No images processed.")

print("Inference completed for all images.")
