import sys
import os

# Add project root to sys.path so `import types` finds our local types.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
