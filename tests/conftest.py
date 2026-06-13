"""Ensure the src/ layout is importable when pytest is run without an editable install."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
