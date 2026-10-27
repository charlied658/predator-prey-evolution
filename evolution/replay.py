"""Replay a saved baseline or study champion."""
import argparse
import subprocess
from pathlib import Path
from config import GODOT_PATH, PROJECT_PATH

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("genome", nargs="?", type=Path, default=Path("best_predator.json"))
    args = parser.parse_args()
    subprocess.run([GODOT_PATH, "--path", PROJECT_PATH, "--", "--replay", str(args.genome.resolve())], check=True)
