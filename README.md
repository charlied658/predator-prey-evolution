# Predator–Prey Evolution

A multi-agent ecosystem with predators, prey, food, hunger, rocks, limited vision,
and home-base respawning. Controllers use explicit hunting, evasion, foraging,
and obstacle-avoidance rules. A small evolutionary algorithm tunes their numbers.

## Setup

Requires Godot 4.7+ and Python 3.10+. Simulation and CSV output use only Python's standard library; plots use Matplotlib.
Open `project.godot` in Godot once to import the project. **F5** runs the ecosystem.

From the repository root, create your local configuration if it does not exist:

```bash
cp evolution/config.json.example evolution/config.json
```

Set the absolute paths to the Godot executable and this repository. For macOS:

```json
{
  "godot_path": "/Applications/Godot.app/Contents/MacOS/Godot",
  "project_path": "/absolute/path/to/predator-prey-evolution"
}
```

The local configuration and generated `results/` are ignored by Git.

## Numeric experiments (main workflow)

Compare mutation-only evolution, a genetic algorithm with crossover, and
elitist differential evolution without opening or rendering the simulation:

```bash
.venv/bin/python evolution/compare_ecosystem.py --output results/my-comparison
```

Create the plotting environment if it is not installed yet:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The default comparison uses 4 candidates, 3 generations, and 10-second episodes.
It saves fitness plots against generation, evaluation budget, and elapsed time,
plus CSV/JSON data. All algorithms share initial populations and scenario seeds.
Training and held-out validation fitness are plotted separately. No fitness
selection is based on the appearance of a run.

For numeric output without plotting dependencies, use
`python3 evolution/compare_ecosystem.py --no-plots --output results/csv-only`.
See [NUMERIC_EXPERIMENTS.md](NUMERIC_EXPERIMENTS.md) for algorithms, paired runs,
budgets, plotting, and interpretation.

## Optional visualization

Watch any saved controller when you want to inspect its behavior:

```bash
python3 evolution/ecosystem.py --config results/my-comparison/de/run-000/best.json
```

Run the original ecosystem with `python3 evolution/ecosystem.py`, or press F5
in Godot. Space pauses; R repeats the seed; N changes it; 1/2/4 changes speed;
Esc exits. [ECOSYSTEM.md](ECOSYSTEM.md) describes its mechanics.

## Run basic evolution

```bash
python3 evolution/evolve_ecosystem.py --output results/my-first-evolution
```

Defaults are four candidate prey controllers, three generations, and 15-second
episodes on one fixed seed. Start from the original behavior, keep the best,
and make mutated copies for the next generation. Add `--role predator` to evolve
predators instead. The other species keeps its starting behavior.

Watch the original and final controllers under the same conditions:

```bash
python3 evolution/ecosystem.py --config results/my-first-evolution/baseline.json
python3 evolution/ecosystem.py --config results/my-first-evolution/best.json
```

`history.csv` records fitness, capture/food/starvation counts, and gene values.
Per-generation configs and every candidate's genes/scores are also saved.
Use a fresh output directory per run. See the [algorithm and fitness formulas](ECOSYSTEM.md#basic-evolutionary-algorithm).

## What is the genome?

The genetic representation is a few named real numbers. A prey genome starts as:

```json
{
  "target_weight": 2.5,
  "avoid_weight": 1.8,
  "wander_weight": 0.5,
  "food_threshold": 85.0
}
```

| Gene | Effect |
| --- | --- |
| `target_weight` | Strength of evasion for prey, or pursuit for predators, relative to avoidance. |
| `avoid_weight` | Strength of steering away from walls, rocks, and nearby animals. |
| `wander_weight` | Strength of wandering relative to avoidance when no target is pursued. |
| `food_threshold` | Prey seek visible food below this energy level when no predator is visible. |

Predators use the first three genes. These genes directly parameterize the
scripted rules; there are no neural weights or learned embeddings. Speed, vision,
body shape, and the environment are fixed settings, not genes in this version.
All animals of the evolving species share one candidate genome during an episode.
Respawning reuses that genome. Selection and mutation happen between generations.

## Verification

```bash
python3 -m unittest discover -s evolution -p 'test_*.py' -v
/Applications/Godot.app/Contents/MacOS/Godot --headless --log-file /tmp/ecosystem-check.log --path . --script tests/ecosystem_smoke.gd
```

The earlier pairwise neural controllers, experiments, and replay code have been
removed from the current tree. They remain available in Git history. Local saved
results are preserved; this version uses ecosystem configuration files.
