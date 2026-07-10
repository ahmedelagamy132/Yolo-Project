# YOLO edge-device benchmarks

This repository contains the experimental code and recorded results used to compare YOLO models on resource-constrained edge devices, including NVIDIA Jetson boards and Raspberry Pi systems.

The benchmark logic is preserved from the original repository revision `830f948`. Files have been renamed and organized, and dataset/log paths were updated for the new structure. The CPU/RAM sampling, latency calculation, disk-I/O behavior, IoU calculation, model selection, and validation calls remain the original research implementation.

See [docs/CODE_PROVENANCE.md](docs/CODE_PROVENANCE.md) for the original-to-organized file mapping and the complete list of permitted changes.

## Repository layout

```text
.
|-- assets/                  Figures used by the project
|-- configs/                 Dataset and device configuration
|-- data/
|   `-- archives/            Compressed sample datasets
|-- docs/                    Benchmark notes and device commands
|-- results/
|   |-- average_precision/   Recorded validation output by YOLO version
|   |-- iou/                 Recorded IoU output
|   `-- logs/
|       `-- reference/       Logs retained from the original repository
`-- scripts/
    |-- benchmarks/          Original experimental programs, renamed
    |-- data/                Original COCO preparation utilities, renamed
    |-- legacy/              Duplicate/test scripts retained for provenance
    |-- reporting/           Original log parser, renamed
    |-- system/              Device/cache helper
    `-- run_benchmarks.sh    Organized equivalent of all_script.sh
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

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

Extract the included COCO128 sample before running the benchmarks:

```bash
unzip data/archives/coco128.zip -d data
```

On Windows PowerShell:

```powershell
Expand-Archive data/archives/coco128.zip -DestinationPath data
```

Run all commands from the repository root. Ultralytics downloads model weights automatically when they are not already available.

## Reproduce the original benchmark flow

The organized runner preserves the original `all_script.sh` order and its YOLO11n model setting:

```bash
bash scripts/run_benchmarks.sh
```

It appends output to `results/logs/generated/yolo11n.log`, matching the original append behavior. Remove or archive that generated log before a clean experimental run.

The runner executes these scripts in order:

1. `benchmark_model_load_time.py`
2. `benchmark_disk_io.py`
3. `benchmark_resources.py`
4. `validate_model.py`
5. `benchmark_iou.py`

The disk-I/O benchmark retains `save=True`, so prediction images are written by Ultralytics and included in the measured write activity.

## Individual scripts

```bash
python scripts/benchmarks/benchmark_disk_io.py
python scripts/benchmarks/benchmark_iou.py
python scripts/benchmarks/benchmark_latency.py
python scripts/benchmarks/benchmark_model_load_time.py
python scripts/benchmarks/benchmark_resources.py
python scripts/benchmarks/run_inference.py
python scripts/benchmarks/validate_model.py
```

The original model-selection behavior is intentionally retained:

- Disk I/O, CPU/RAM, IoU, and validation read `YOLO_MODEL_SIZE`, defaulting to `11n`.
- The standalone latency script uses `yolo11s.pt`.
- The standalone inference script uses `yolo11m.pt`.
- The model-load script uses the original `YOLO("yolo11n")` call.

## Data and reporting utilities

The original data-preparation algorithms are retained. Their path variables now point to `data/coco`; edit those variables if the full COCO dataset is stored elsewhere.

```bash
python scripts/data/create_validation_list.py
python scripts/data/convert_coco_to_yolo.py
python scripts/reporting/extract_metrics.py
```

The COCO-to-YOLO converter retains the original append mode. Running it more than once without clearing the target label directory will duplicate annotations.

## Research reproducibility notes

To compare a new run with the recorded results, use the same:

- device and power configuration;
- operating system and drivers;
- Python and package versions;
- YOLO weight files;
- COCO data files and image order;
- background workload and thermal conditions.

Timing, CPU, RAM, and disk measurements can still vary naturally between runs even when the code is identical.

The only intentional non-path safety difference is `scripts/system/clear_cache.sh`: it is not called by the benchmark runner, and its Python-cache deletion is restricted to this repository rather than the entire computer.

FPS is calculated as `1000 / total latency in milliseconds`; see [docs/fps_calculation.md](docs/fps_calculation.md). The empty `results/average_precision/v11/yolo11m.txt` file is retained because it was empty in the original repository.
