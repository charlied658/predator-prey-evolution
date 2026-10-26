import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

GODOT_PATH = config["godot_path"]
PROJECT_PATH = config["project_path"]