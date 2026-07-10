# Legacy scripts

These files preserve duplicate or test scripts that existed in the original repository revision `830f948`:

- `benchmark_resources_yolo11x.py` was `ram&cpu_test.py`.
- `validate_model_cpu.py` was `test.py`.
- `run_all_yolo11x.sh` was `run_all.sh`.

Only paths and filenames were adapted to the organized repository structure. The legacy runner retains the original behavior of invoking every listed file with `python3`, including the shell cache script; this is preserved for provenance and is not the recommended runner.
