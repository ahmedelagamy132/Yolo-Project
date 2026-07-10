#!/bin/bash
# Runs the standard sequence of YOLO benchmark scripts and appends their output to one log.

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

output_file="results/logs/generated/yolo11n.log"
mkdir -p "$(dirname "$output_file")"

scripts=("scripts/benchmarks/benchmark_model_load_time.py" "scripts/benchmarks/benchmark_disk_io.py" "scripts/benchmarks/benchmark_resources.py" "scripts/benchmarks/validate_model.py" "scripts/benchmarks/benchmark_iou.py")

model_size="11n"

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
