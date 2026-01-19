import sys
from pathlib import Path

# Ensure project root is always first on sys.path so sibling modules import reliably
PROJECT_ROOT = Path(__file__).resolve().parents[1]
root_str = str(PROJECT_ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)
