"""Optional local Godot settings; the Python pursuit viewer needs no Godot."""
import json
import shutil
from src.paths import ROOT, CONFIGS

CONFIG_PATH = CONFIGS / "config.json"
if not CONFIG_PATH.exists():
    CONFIG_PATH = ROOT / "evolution" / "config.json"  # Existing local setups.
config = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
GODOT_PATH = config.get("godot_path") or shutil.which("godot") or shutil.which("godot4")
if not GODOT_PATH:
    GODOT_PATH = "/Applications/Godot.app/Contents/MacOS/Godot"
PROJECT_PATH = config.get("project_path", str(ROOT))
