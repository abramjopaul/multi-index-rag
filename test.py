#!/usr/bin/env python3
import subprocess
import sys

print("=" * 60)
print("CUDA Availability Check")
print("=" * 60)

# Check 1: nvidia-smi
print("\n1. Checking nvidia-smi...")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("✓ nvidia-smi found:")
        print(result.stdout[:500])  # Print first 500 chars
    else:
        print("✗ nvidia-smi failed")
except FileNotFoundError:
    print("✗ nvidia-smi not found in PATH")
except Exception as e:
    print(f"✗ Error: {e}")

# Check 2: PyTorch
print("\n2. Checking PyTorch CUDA support...")
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - Device {i}: {torch.cuda.get_device_name(i)}")
except ImportError:
    print("✗ PyTorch not installed")
except Exception as e:
    print(f"✗ Error: {e}")

# Check 3: TensorFlow
print("\n3. Checking TensorFlow CUDA support...")
try:
    import tensorflow as tf
    print(f"✓ TensorFlow version: {tf.__version__}")
    print(f"✓ CUDA available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU count: {len(gpus)}")
        for gpu in gpus:
            print(f"  - {gpu}")
except ImportError:
    print("✗ TensorFlow not installed")
except Exception as e:
    print(f"✗ Error: {e}")

# Check 4: cuDNN
print("\n4. Checking cuDNN...")
try:
    from torch.backends import cudnn
    if cudnn.is_available():
        print(f"✓ cuDNN available: {cudnn.version()}")
    else:
        print("✗ cuDNN not available")
except ImportError:
    print("✗ PyTorch not installed")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 60)
