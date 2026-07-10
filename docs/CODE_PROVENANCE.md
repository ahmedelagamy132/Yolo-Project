# Code provenance

The experimental logic in this repository was restored from original repository revision `830f948` (`Update all_script.sh`).

The table records where each original executable file now lives. Renames and paths were changed only to organize the project and allow commands to run from the repository root.

| Original path | Organized path |
| --- | --- |
| `Final/disk5.py` | `scripts/benchmarks/benchmark_disk_io.py` |
| `Final/iou.py` | `scripts/benchmarks/benchmark_iou.py` |
| `Final/latency.py` | `scripts/benchmarks/benchmark_latency.py` |
| `Final/load_time.py` | `scripts/benchmarks/benchmark_model_load_time.py` |
| `Final/ram&cpu.py` | `scripts/benchmarks/benchmark_resources.py` |
| `Final/running_coco128.py` | `scripts/benchmarks/run_inference.py` |
| `Final/validation.py` | `scripts/benchmarks/validate_model.py` |
| `Final/gpu.py` | `scripts/benchmarks/check_gpu.py` |
| `Final/test_cpu.py` | `scripts/benchmarks/check_system_resources.py` |
| `Final/script.py` | `scripts/data/create_validation_list.py` |
| `Final/script2.py` | `scripts/data/convert_coco_to_yolo.py` |
| `Final/extraction.py` | `scripts/reporting/extract_metrics.py` |
| `Final/all_script.sh` | `scripts/run_benchmarks.sh` |
| `Final/ram&cpu_test.py` | `scripts/legacy/benchmark_resources_yolo11x.py` |
| `Final/test.py` | `scripts/legacy/validate_model_cpu.py` |
| `Final/run_all.sh` | `scripts/legacy/run_all_yolo11x.sh` |

The permitted organizational changes are:

- relative dataset paths now begin with `data/`;
- the validation scripts reference `configs/coco128.yaml`;
- generated logs are stored under `results/logs/generated/`;
- the organized runners change to the repository root and create the generated-log directory before running;
- empty spaces at line ends were removed.

`scripts/system/clear_cache.sh` is the only behaviorally changed helper. It is not called by the standard runner. Its Python-cache cleanup is intentionally confined to this repository rather than deleting caches across the entire computer.
