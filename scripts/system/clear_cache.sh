#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ "$(uname -s)" == "Linux" && -w /proc/sys/vm/drop_caches ]]; then
    echo "Clearing Linux file-system caches..."
    sync
    echo 3 > /proc/sys/vm/drop_caches
else
    echo "Skipping system cache: run as root on Linux to write /proc/sys/vm/drop_caches."
fi

echo "Removing Python bytecode caches inside the repository..."
find "$root_dir" -type d -name __pycache__ -prune -exec rm -rf {} +

python3 - <<'PY'
try:
    import torch
    torch.cuda.empty_cache()
    print("Cleared the PyTorch CUDA cache.")
except (ImportError, RuntimeError) as error:
    print(f"Skipping the PyTorch CUDA cache: {error}")

try:
    import tensorflow as tf
    tf.keras.backend.clear_session()
    print("Cleared the TensorFlow session.")
except (ImportError, RuntimeError) as error:
    print(f"Skipping the TensorFlow session: {error}")
PY
