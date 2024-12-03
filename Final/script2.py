import json
import os

# Paths to COCO annotations and directories
coco_json_path = r"C:\Users\ahmed\Desktop\Yolo-Project\accuracy\datasets\coco\annotations\instances_val2017.json"
images_dir = r"C:\Users\ahmed\Desktop\Yolo-Project\accuracy\datasets\coco\val2017"
labels_dir = r"C:\Users\ahmed\Desktop\Yolo-Project\accuracy\datasets\coco\labels\val2017"

# Create labels directory if it doesn't exist
os.makedirs(labels_dir, exist_ok=True)

# Load COCO JSON file
with open(coco_json_path, "r") as f:
    coco_data = json.load(f)

# Map COCO category IDs to continuous class IDs
categories = {cat["id"]: idx for idx, cat in enumerate(coco_data["categories"])}

# Iterate over annotations and generate YOLO labels
for ann in coco_data["annotations"]:
    image_id = ann["image_id"]
    category_id = ann["category_id"]
    bbox = ann["bbox"]  # COCO format: [x_min, y_min, width, height]

    # Convert bbox to YOLO format
    x_min, y_min, width, height = bbox
    x_center = x_min + width / 2
    y_center = y_min + height / 2

    # Get image dimensions
    image_info = next((img for img in coco_data["images"] if img["id"] == image_id), None)
    if not image_info:
        print(f"Image ID {image_id} not found in images.")
        continue

    img_width = image_info["width"]
    img_height = image_info["height"]

    # Normalize bbox coordinates
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height

    # Prepare YOLO label line
    label_line = f"{categories[category_id]} {x_center} {y_center} {width} {height}\n"

    # Write to corresponding label file
    label_file_path = os.path.join(labels_dir, f"{image_id:012d}.txt")
    with open(label_file_path, "a") as label_file:
        label_file.write(label_line)

print(f"YOLO labels generated in {labels_dir}")
