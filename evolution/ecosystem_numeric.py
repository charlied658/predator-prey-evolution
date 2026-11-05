"""Batched headless evaluation; numeric and visible episodes share Godot simulation code."""
import json
import math
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config import GODOT_PATH, PROJECT_PATH


def _batch(settings):
    with tempfile.TemporaryDirectory(prefix='ecosystem-numeric-') as directory:
        folder = Path(directory)
        source, target = folder / 'jobs.json', folder / 'results.json'
        source.write_text(json.dumps(settings, allow_nan=False), encoding='utf-8')
        command = [GODOT_PATH, '--headless', '--path', PROJECT_PATH,
                   '--log-file', str(folder / 'godot.log'), '--script', 'res://scripts/ecosystem_numeric.gd',
                   '--', str(source), str(target)]
        process = subprocess.run(command, capture_output=True, text=True,
                                 timeout=max(60, sum(job['duration'] for job in settings) * 5))
        if process.returncode or not target.exists():
            raise RuntimeError(f'Numeric evaluation failed ({process.returncode}):\n{process.stdout}\n{process.stderr}')
        rows = json.loads(target.read_text())
        if len(rows) != len(settings):
            raise RuntimeError('Numeric evaluation returned an incomplete batch')
        return rows


def evaluate(settings, workers=1):
    if workers < 1:
        raise ValueError('workers must be positive')
    if not settings:
        return []
    width = math.ceil(len(settings) / min(workers, len(settings)))
    batches = [settings[i:i + width] for i in range(0, len(settings), width)]
    if len(batches) == 1:
        return _batch(batches[0])
    with ThreadPoolExecutor(max_workers=len(batches)) as pool:
        return [row for rows in pool.map(_batch, batches) for row in rows]
