# YOLO Edge-Device Benchmarks

This project benchmarks YOLO object-detection models on edge computing devices. It provides scripts for measuring model load time, inference latency, FPS, disk activity, CPU use, memory use, validation accuracy, and bounding-box IoU.

The repository also includes sample COCO128 data, benchmark logs, average-precision outputs, configuration files, and supporting documentation for running experiments on devices such as NVIDIA Jetson boards and Raspberry Pi systems.

## Project layout

```text
assets/                  Project figures
configs/                 Dataset and device configuration files
data/                    Dataset archives and local data files
docs/                    Setup and measurement notes
results/                 Benchmark logs and recorded results
scripts/                 Benchmark, data-preparation, and utility scripts
```

## Setup

```bash
git clone https://github.com/ahmedelagamy132/Yolo-Project.git
cd Yolo-Project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Extract the included COCO128 sample before running image benchmarks:

```bash
unzip data/archives/coco128.zip -d data
```

On Windows PowerShell:

```powershell
Expand-Archive data/archives/coco128.zip -DestinationPath data
```

## Run benchmarks

Run the complete benchmark suite from the repository root:

```bash
bash scripts/run_benchmarks.sh
```

Run individual benchmarks as needed:

```bash
python scripts/benchmarks/benchmark_disk_io.py
python scripts/benchmarks/benchmark_latency.py
python scripts/benchmarks/benchmark_resources.py
python scripts/benchmarks/benchmark_iou.py
python scripts/benchmarks/validate_model.py
```

Generated logs are saved under `results/logs/generated/`. Additional benchmark variants and supporting utilities are available under `scripts/legacy`, `scripts/data`, and `scripts/reporting`.

## Notes

- Model weights are downloaded automatically by Ultralytics when required.
- Benchmark results can vary with hardware, operating-system configuration, temperature, power mode, and background load.
- FPS is calculated from total image-processing latency; see [docs/fps_calculation.md](docs/fps_calculation.md).
