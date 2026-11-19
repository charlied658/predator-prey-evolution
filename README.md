# Predator–Prey Evolution

Train predator and prey movement controllers, then explore their behavior in a
live interactive viewer. One predator pursues multiple prey that forage, evade,
and respawn. Evolution changes movement controllers; target priorities, sensing,
and collision rules are programmed. The viewer uses saved controllers and does
not train while playing.

![example1](assets/example1.png)

## Setup

Run all commands from the repository root. Use Python 3.10 or newer with tkinter
support. The current Python viewer and training workflow do not require Godot.

**Windows — PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**macOS — Terminal**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

If you already have `.venv`, skip creating it. To check tkinter, run your virtual
environment's Python with `-m tkinter`; a small test window should open. If tkinter
is missing, install a Python distribution with Tk support and recreate `.venv`.

## Generate a sample and open the viewer

Saved runs live in `results/`, which Git ignores. A fresh clone or a pull on
another computer does not bring trained controllers with it. Generate this small
sample, or copy a compatible saved run from your other computer.

The sample uses an empty arena, one predator, three prey, and two generations.
It checks the complete workflow without needing a terrain file.

**Windows — PowerShell**

```powershell
.\.venv\Scripts\python.exe -u -m src.training --arena empty --respawn --prey-count 3 --generations 2 --seeds 1 --training-starts 1 --test-starts 2 --seconds 10 --replay-seconds 10 --output results/viewer-sample
.\.venv\Scripts\python.exe -m src.viewer results/viewer-sample
```

**macOS — Terminal**

```bash
.venv/bin/python -u -m src.training --arena empty --respawn --prey-count 3 --generations 2 --seeds 1 --training-starts 1 --test-starts 2 --seconds 10 --replay-seconds 10 --output results/viewer-sample
.venv/bin/python -m src.viewer results/viewer-sample
```

Wait for training to finish before opening the viewer. Training requires a new
output directory; use another name if `results/viewer-sample` exists. Reopening
the viewer uses the existing run and does not retrain.

For the commands below, `python` means your virtual environment's Python:
`.venv/bin/python` on macOS or `.\.venv\Scripts\python.exe` on Windows.

## Use the interactive viewer

```bash
python -m src.viewer results/viewer-sample --paused
```

- **Run:** choose a controller seed and saved generation, encounter seed, prey
  count, and duration. Restart repeats an encounter; New encounter changes its
  seed. Space plays or pauses; Right Arrow steps while paused.
- **Map & food:** select the training map or empty arena, generate terrain, or
  load an exported 96 × 96 terrain JSON. Changing maps restarts the encounter.
- **Overlays:** select one organism from the dropdown or click its body.
  Checkboxes show only its vision, route waypoints, final target, visited squares,
  trail, and label. Switching selection moves the indicators while preserving
  checkbox choices. Food and grid controls apply to the whole arena.

Playback speed and overlays do not change simulation physics. For comparisons,
keep the map, encounter seed, prey count, and other simulation settings fixed.
A controller trained in an empty arena may perform differently on complex terrain.

Omit the run argument to load the latest compatible run directly inside
`results/`. To transfer a run, copy its entire directory, including
`champions.json`, `instructions.json`, `terrain.json`, and `history.json` if
available. Older ecosystem comparison results are a different format and cannot
be opened in this viewer.

## Train a longer experiment

This starts fresh with 30 generations and three independent training seeds.
Each encounter lasts 30 simulated seconds. Training runs without the viewer;
progress prints in the terminal. Total runtime depends on your machine and settings.

```bash
python -u -m src.training --arena empty --respawn --prey-count 3 --generations 30 --seeds 3 --training-starts 4 --test-starts 16 --seconds 30 --replay-seconds 60 --eval-every 5 --output results/pursuit-30gen
python -m src.viewer results/pursuit-30gen --paused
```

More generations provide more opportunities for evolution, but do not guarantee
improvement. Compare saved generations in the viewer and inspect `outcomes.png`
and `measurements.csv`. Use `python -m src.training --help` for all options.
For terrain training, use `--arena terrain --terrain-map PATH_TO_TERRAIN.json`.

## Tests

```bash
python -m unittest discover -s tests/python -v
```

The viewer test briefly creates a desktop window, so run it in a desktop session.
Tests for the older Godot ecosystem also require Godot. If it is not found
on your system, copy `src/configs/config.json.example` to
`src/configs/config.json` and set the executable and project paths. Existing
`evolution/config.json` settings remain supported.

## Project layout

| Directory | Contents |
| --- | --- |
| `src/viewer/` | Interactive viewer |
| `src/training/` | Current pursuit training and basic ecosystem evolution |
| `src/simulation/` | Simulation, controllers, and shared foraging routines |
| `src/terrain/` | Terrain generation, collision, navigation, and preview launcher |
| `src/analysis/` | Evaluation, plots, replay rendering, and shared font loading |
| `src/experiments/` | Earlier experiment entry points |
| `src/configs/` | Default rules and optional local Godot configuration |
| `tests/python/`, `tests/godot/` | Python and Godot tests |
| `scripts/`, `*.tscn` | Godot scripts and scenes |
| `docs/` | Historical experiments and older workflow guides |
| `assets/` | Documentation images |
| `results/` | Generated runs, ignored by Git |

The old `evolution/pursuit_viewer.py` and `evolution/hybrid_pursuit.py` commands
remain as compatibility shortcuts. Shared vision and foraging modules retain
older experiment entry points alongside their simulation functions.

Historical workflows and development notes are preserved in
[Historical experiments](docs/HISTORICAL_EXPERIMENTS.md),
[Godot ecosystem](docs/ECOSYSTEM.md), and
[Numeric comparisons](docs/NUMERIC_EXPERIMENTS.md).
