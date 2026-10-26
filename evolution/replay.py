import subprocess
from pathlib import Path
from config import GODOT_PATH, PROJECT_PATH

GENOME_PATH = Path("best_predator.json").resolve()

subprocess.run([
    GODOT_PATH,
    "--path", PROJECT_PATH,
    "--",
    "--replay", str(GENOME_PATH)
])