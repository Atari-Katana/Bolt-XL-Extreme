import sys
import os

# Add root to sys.path
sys.path.append(os.getcwd())

try:
    import tests.kernels.moe.utils
    print("Import success")
except ImportError as e:
    print(f"Import failed: {e}")
