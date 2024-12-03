import os
from ultralytics import YOLO

model = YOLO("yolo11m.pt")

image_folder = r"coco128/images/train2017"

for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        print(f"Processing {image_name}...")
        
        results = model(image_path, save=False, show=False)

print("Inference completed for all images.")