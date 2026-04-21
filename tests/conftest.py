import sys
import os

# Add src/ to sys.path so `import coder` works without `pip install -e .`
_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_root, "src"))
# Also keep project root so cross-test imports (e.g. from tests.X import ...) resolve
sys.path.insert(0, _root)
