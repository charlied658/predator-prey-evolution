import subprocess
import json
from pathlib import Path
from config import GODOT_PATH, PROJECT_PATH

INPUT_PATH = Path("batch_input.json").resolve()
OUTPUT_PATH = Path("batch_results.json").resolve()

def evaluate_batch(encounters):
    with open(INPUT_PATH, "w") as f:
        json.dump({"encounters": encounters}, f, indent=2)

    result = subprocess.run(
        [
            GODOT_PATH,
            "--headless",
            "--path", PROJECT_PATH,
            "--fixed-fps", "60",
            "--",
            "--batch",
            str(INPUT_PATH),
            str(OUTPUT_PATH),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("Godot batch evaluation failed.")

    with open(OUTPUT_PATH, "r") as f:
        return json.load(f)["results"]