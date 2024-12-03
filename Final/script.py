import json
import os

# Path to the JSON annotation file
json_path = r"C:\Users\ahmed\Desktop\Yolo-Project\accuracy\datasets\coco\annotations\instances_val2017.json"

# Path to the directory containing validation images
image_dir = r"C:\Users\ahmed\Desktop\Yolo-Project\accuracy\datasets\coco\images\val2017"

# Output file to store image paths
output_txt = r"C:\Users\ahmed\Desktop\Yolo-Project\datasets\coco\val2017.txt"

# Load the JSON file
with open(json_path, "r") as f:
    data = json.load(f)

# Extract image file names and create paths
image_paths = []
for image in data["images"]:
    image_name = image["file_name"]
    full_path = os.path.join(image_dir, image_name)
    if os.path.exists(full_path):  # Ensure the file exists
        image_paths.append(full_path)
    else:
        print(f"Missing file: {full_path}")

# Write paths to the text file
with open(output_txt, "w") as f:
    f.writelines("\n".join(image_paths))

print(f"Validation file list saved to {output_txt}")