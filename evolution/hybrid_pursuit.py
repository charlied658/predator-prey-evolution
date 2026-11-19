"""Compatibility shortcut; new entry point: python -m src.training.hybrid_pursuit."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.hybrid_pursuit import main

if __name__ == '__main__':
    main()
