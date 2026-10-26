# Predator–Prey Evolution

A Godot + Python simulation for experimenting with evolutionary neural-network controllers in predator–prey encounters. Godot runs the encounters; Python evaluates fitness and evolves predator controllers using a genetic algorithm.

## Requirements

- **Godot 4.x** (the executable, not just the project editor)
- **Python 3.10+**
- Windows, macOS, or Linux

The current Python scripts use only the standard library; no `pip install` is required.

## Project layout

The expected layout is:

```text
PredatorPreyEvolution/
├── godot/                  # Godot project (contains project.godot)
├── evolution/
│   ├── config.py
│   ├── config.example.json
│   ├── config.json         # Local paths; do not commit
│   ├── godot_interface.py
│   ├── genetic_algorithm.py
│   ├── replay.py
│   ├── test_batch.py
│   └── test_generalization.py
└── .gitignore
```

Your repository may use different folder names. The important part is that `project_path` points to the folder containing **`project.godot`**.

## 1. Get the project

Clone the repository and open it in your editor:

```bash
git clone <REPOSITORY_URL>
cd PredatorPreyEvolution
```

Replace `<REPOSITORY_URL>` with the actual GitHub repository URL.

## 2. Configure local paths

Copy the example configuration:

**PowerShell (Windows):**

```powershell
Copy-Item evolution/config.example.json evolution/config.json
```

**macOS / Linux:**

```bash
cp evolution/config.example.json evolution/config.json
```

Edit `evolution/config.json`:

```json
{
  "godot_path": "C:/path/to/Godot.exe",
  "project_path": "C:/path/to/PredatorPreyEvolution/godot"
}
```

- `godot_path`: full path to your Godot executable.
- `project_path`: full path to the Godot project directory (the folder containing `project.godot`).
- On Windows, forward slashes work well in JSON paths.
- `config.json` is machine-specific and should be excluded by `.gitignore`; **do not commit it**.

## 3. Check the Godot project

Open the Godot project in the editor and confirm that `world.tscn` is configured as the main scene (**Project → Project Settings → Application → Run**). The world contains a predator and prey, both controlled by neural networks.

The batch simulation reads genome arrays from JSON and writes encounter results to JSON. The Python interface starts Godot in headless mode to run these encounters quickly.

## 4. Run a batch test

From the repository root:

```bash
cd evolution
python test_batch.py
```

On systems where `python` is not available, try `python3` or `py`.

A successful test prints encounter results with fields such as `captured`, `time`, `predator_fitness`, and `prey_fitness`.

## 5. Train a predator

From the `evolution` directory:

```bash
python genetic_algorithm.py
```

The current genetic algorithm:

- Evolves a population of predator neural-network genomes (66 parameters each).
- Evaluates them against one fixed prey genome.
- Uses tournament selection, uniform crossover, Gaussian mutation, and elitism.
- Prints best and average fitness for each generation.
- Saves the best predator and the training prey to **`best_predator.json`**.

Training settings such as population size, number of generations, and mutation rate are defined near the top of `genetic_algorithm.py`.

**Note:** For replay and generalization testing, the saved JSON should include both `genome` and `prey_genome`, along with `fitness_history`.

## 6. Watch the evolved behavior

After training, run:

```bash
python replay.py
```

This launches Godot with a visible window and replays the best predator against its training prey. The encounter ends after capture or when the time limit is reached.

## 7. Test generalization

To test the evolved predator against 20 newly generated prey controllers:

```bash
python test_generalization.py
```

The script reports the result against the original training prey, the capture rate against random prey, and average encounter duration. This helps distinguish improvement against a specific opponent from broader pursuit ability.

## How fitness works

Each encounter lasts up to **20 simulated seconds**. Capture occurs when the agents come within **1 unit** of each other.

- **Predator:** receives `100 + (20 - capture_time)` on capture, otherwise `0`.
- **Prey:** receives the number of seconds it survives.

Both agents use fixed-topology neural networks with **5 inputs, 8 hidden units, and 2 movement outputs**. Their movement is learned from genome parameters rather than hardcoded chase/flee rules.

## Troubleshooting

**Godot executable not found:** Check `godot_path` in `evolution/config.json` and confirm the file exists.

**Godot cannot find the project:** Check that `project_path` points to the directory containing `project.godot`.

**`FileNotFoundError: config.json`:** Create `evolution/config.json` from the example configuration.

**Replay cannot load `prey_genome`:** Rerun training with the updated save logic that includes the prey genome.

**Batch evaluation fails:** Open the Godot project in the editor to check for GDScript errors, and inspect the stdout/stderr printed by the Python interface.

**Generated files appear in Git:** Make sure `.gitignore` excludes `evolution/config.json`, Python caches, and Godot's `.godot/` directory. You may also choose to ignore training outputs such as `best_predator.json` and batch input/output JSON files.

## Current limitations

This is an early experimental baseline, not yet a full coevolution system. Training uses a single fixed prey opponent and fixed initial positions, so successful behavior may be specialized to that setup. Planned extensions include training against multiple opponents, randomized starting positions, and evolving predator and prey populations together.
