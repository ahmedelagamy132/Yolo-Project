#!/bin/bash

output_file="logs/test.log"

scripts=("load_time.py" "disk5.py" "ram&cpu.py" "validation.py" "iou.py" "clear_cache.sh")

model_size="11x"  

for script in "${scripts[@]}"; do
    echo "Running $script..." | tee -a "$output_file"
    YOLO_MODEL_SIZE="$model_size" python3 "$script" >> "$output_file" 2>&1
    if [ $? -eq 0 ]; then
        echo "$script executed successfully." | tee -a "$output_file"
    else
        echo "Error occurred while running $script." | tee -a "$output_file"
    fi
    echo "----------------------------------------" >> "$output_file"
done

echo "All scripts executed. Results are saved in $output_file."