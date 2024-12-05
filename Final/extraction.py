import re
import os
import json

def extract_specific_lines(file_path):
    """
    Extracts specific lines from a .log file based on predefined patterns.

    Args:
        file_path (str): Path to the .log file.

    Returns:
        dict: Extracted information categorized by line types.
    """
    results = {
        "Average CPU Usage": [],
        "Average Memory Usage": [],
        "YOLO Summary": [],
        "Performance Metrics": [],
        "Model Load Time": [],
        "Average Read Bytes": [],
        "Average Write Bytes": [],
        "Average Disk Read Speed": [],
        "Average Disk Write Speed": [],
        "Average Latency": [],
        "FPS": [],
        "Average IoU": []
    }

    # Define regex patterns
    cpu_pattern = r"^Average CPU Usage \(script-specific\):\s*(.*)"
    memory_pattern = r"^Average Memory Usage \(script-specific\):\s*(.*)"
    yolo_pattern = r"^YOLO\d+[a-z] summary \(fused\):\s*(.*)"
    metrics_pattern = r"^\s+all\s+(\d+)\s+(\d+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)"
    load_time_pattern = r"^Model load time:\s*(.*)"
    read_bytes_pattern = r"^Average Read Bytes:\s*(.*)"
    write_bytes_pattern = r"^Average Write Bytes:\s*(.*)"
    read_speed_pattern = r"^Average Disk Read Speed:\s*(.*)"
    write_speed_pattern = r"^Average Disk Write Speed:\s*(.*)"
    latency_pattern = r"^Average latency:\s*(.*)"
    fps_pattern = r"^FPS:\s*(.*)"
    iou_pattern = r"^Average IoU across all images:\s*(.*)"

    with open(file_path, 'r', encoding='utf-8') as log_file:
        for line in log_file:
            if match := re.match(cpu_pattern, line):
                results["Average CPU Usage"].append(match.group(1))
            elif match := re.match(memory_pattern, line):
                results["Average Memory Usage"].append(match.group(1))
            elif match := re.match(yolo_pattern, line):
                results["YOLO Summary"].append(match.group(1))
            elif match := re.match(metrics_pattern, line):
                performance_data = {
                    "images": int(match.group(1)),
                    "instances": int(match.group(2)),
                    "precision": float(match.group(3)),
                    "recall": float(match.group(4)),
                    "mAP50": float(match.group(5)),
                    "mAP50-95": float(match.group(6))
                }
                results["Performance Metrics"].append(performance_data)
            elif match := re.match(load_time_pattern, line):
                results["Model Load Time"].append(match.group(1))
            elif match := re.match(read_bytes_pattern, line):
                results["Average Read Bytes"].append(match.group(1))
            elif match := re.match(write_bytes_pattern, line):
                results["Average Write Bytes"].append(match.group(1))
            elif match := re.match(read_speed_pattern, line):
                results["Average Disk Read Speed"].append(match.group(1))
            elif match := re.match(write_speed_pattern, line):
                results["Average Disk Write Speed"].append(match.group(1))
            elif match := re.match(latency_pattern, line):
                results["Average Latency"].append(match.group(1))
            elif match := re.match(fps_pattern, line):
                results["FPS"].append(match.group(1))
            elif match := re.match(iou_pattern, line):
                results["Average IoU"].append(match.group(1))

    return results

def save_results_to_json(results, output_folder="results", output_file="extracted_data.json"):
    """
    Saves extracted results to a JSON file in a specified folder.

    Args:
        results (dict): Extracted data.
        output_folder (str): Folder to save the file.
        output_file (str): File name to save the results.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Construct the file path
    file_path = os.path.join(output_folder, output_file)

    # Save results to a JSON file
    with open(file_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(results, jsonfile, indent=4)

    print(f"Results saved to {file_path}")

# Example Usage
log_file_path = "logs/11nano.log"  # Replace with your .log file path
extracted_data = extract_specific_lines(log_file_path)
save_results_to_json(extracted_data, output_file="yolo11n.json")
