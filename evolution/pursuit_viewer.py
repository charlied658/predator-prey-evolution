"""Compatibility shortcut; new entry point: python -m src.viewer.pursuit_viewer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.viewer.pursuit_viewer import main

if __name__ == '__main__':
    main()
