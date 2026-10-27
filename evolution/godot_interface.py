"""Isolated, deterministic Godot batch workers (standard library only)."""
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from config import GODOT_PATH, PROJECT_PATH


def _evaluate(encounters):
    with tempfile.TemporaryDirectory(prefix="predator-prey-") as directory:
        input_path = Path(directory) / "input.json"
        output_path = Path(directory) / "output.json"
        input_path.write_text(json.dumps({"encounters": encounters}), encoding="utf-8")
        process = subprocess.run(
            [GODOT_PATH, "--headless", "--path", PROJECT_PATH,
             "--fixed-fps", "60", "--", "--batch", str(input_path), str(output_path)],
            capture_output=True, text=True, timeout=max(60, len(encounters) * 10),
        )
        if process.returncode or not output_path.exists():
            raise RuntimeError(f"Godot failed ({process.returncode}):\n{process.stdout}\n{process.stderr}")
        results = json.loads(output_path.read_text(encoding="utf-8"))["results"]
        if len(results) != len(encounters):
            raise RuntimeError("Godot returned an incomplete batch")
        return results


def evaluate_batch(encounters, workers=1):
    if not encounters:
        return []
    if workers < 1:
        raise ValueError("workers must be positive")
    workers = min(workers, len(encounters))
    size = (len(encounters) + workers - 1) // workers
    chunks = [encounters[i:i + size] for i in range(0, len(encounters), size)]
    if workers == 1:
        return _evaluate(encounters)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = [row for chunk in pool.map(_evaluate, chunks) for row in chunk]
    for index, row in enumerate(results):
        row["encounter"] = index
    return results
