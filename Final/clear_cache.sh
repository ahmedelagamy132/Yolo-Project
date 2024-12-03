#!/bin/bash

echo "Clearing file system cache..."
sudo sync
echo 3 | sudo tee /proc/sys/vm/drop_caches
echo "File system cache cleared."

echo "Clearing Python cache..."
find / -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
echo "Python cache cleared."

clear_gpu_cache() {
    python3 -c "
import torch
import tensorflow as tf

try:
    print('Clearing PyTorch GPU cache...')
    torch.cuda.empty_cache()
    print('PyTorch GPU cache cleared.')
except Exception as e:
    print('PyTorch not in use or error:', e)

try:
    print('Clearing TensorFlow GPU cache...')
    tf.keras.backend.clear_session()
    print('TensorFlow GPU cache cleared.')
except Exception as e:
    print('TensorFlow not in use or error:', e)
"
}

echo "Attempting to clear GPU cache..."
clear_gpu_cache
echo "GPU cache cleared if applicable."

echo "All caches cleared successfully."
