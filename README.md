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

## Co-evolution experiments

### Bit-string co-evolution

This standalone experiment compares fixed, current, and archived opponents in
a mathematical game. It runs in Python without Godot. From PowerShell:

```powershell
& .\.venv\Scripts\python.exe -u evolution/coevolve_bits.py --output results/coevolve-bits
```

Each side mutates an 8-bit genome by flipping one bit per offspring. The payoff
blends lookup tables for each genome with tables that read bits from both sides.
The same 20 seeded landscapes are tested at two interaction weights, 0.2 and
0.8. All three methods start from the same genomes and get 2,560 training
matchups per run. In generation one, all methods use the same panel. The archive
method fills missing historical slots with current opponents until two earlier
checkpoints from training are available.

Every five generations, the runner scores each champion against all 256 possible
opponents. The pure best-response gap is the best payoff A could obtain against
the current B minus the worst payoff A could receive against any B. Smaller
values indicate a more robust pair of pure strategies. A game whose equilibrium
requires a mixture can still have a positive gap for every pure pair.

`progress.png` shows the mean gap over time. `historical_matchups.png` shows
cross-generation payoffs on the first landscape for each condition. CSV files
store the measurements for every landscape, and `summary.json` reports the
paired archive-versus-current comparison. Use `--no-plots` to run with only the
Python standard library, or `--repeats 2 --generations 10` for a quick preview.

### Symmetry breaking

The symmetry experiment starts A and B with identical populations and gives
them independent mutation and opponent sampling. Swapping the genomes reverses
the payoff exactly. Both sides train against current opponents.

```powershell
& .\.venv\Scripts\python.exe -u evolution/coevolve_symmetry.py
```

The default is 20 paired landscapes and 60 generations at each interaction
strength. The runner saves `symmetry.png` for the first landscape and
`history.csv` and `summary.json` for all runs. It tracks genome differences,
differences in payoff against the same full set of opponents, and who wins the
current head-to-head matchup. A reversal counts when the next non-tied winner
is the other side. A tie ends an unbroken lead.

### Symmetric food competition

Run two initially identical populations, with one agent per lineage in
each episode. They compete for 28 scattered pellets among 14 circular rocks.
Each pellet can be collected once. Contact has no scoring effect. Episodes
end at the time limit.

```powershell
& .\.venv\Scripts\python.exe -u evolution/coevolve_foraging.py
```

A 70-weight neural controller maps local observations and a two-channel clock
to steering and joint bend. The clock supplies timing information. Only actual
tail motion generates thrust, and drag slows the body between strokes.
Steering becomes more effective as the body gains speed. This is a simplified
thrust and drag model.

Four body genes set front and rear segment length, their width
ratio, and the bend limit. Both segments collide with terrain. The sum of
their capsule areas before overlap is fixed, and only the head collects
pellets. Both lineages start with the same body proportions.
Each lineage selects by its pellet count minus its opponent's, then makes
mutated copies of its champion. Candidate pairs play from both starting
positions. Every generation, the champions also play each other on a held-out
map and forage individually from both starts on two held-out maps.

The main Godot scene launches the populated ecosystem. The visual launcher
opens the appropriate scene when given a saved experiment configuration.

`specialization.png` shows pellet counts in direct matchups and solo tests,
plus movement speed and path turning in the solo tests. `history.csv` records
each trial, and `genome_history.json` tracks divergence between the champions'
weights. Look for behavioral differences that persist across generations
and matched conditions. This measures champion behavior, while reproduction
between the lineages is outside the experiment. The default uses one
optimizer seed. Use `--seed` and a new output directory for independent runs.
The final body-swap test runs both controllers in both bodies under matched
solo conditions. See `body_swap.csv` and the four `controller-*-body-*.json`
replay configurations.

Replay a matchup or its swapped-start counterpart:

```powershell
& .\.venv\Scripts\python.exe evolution/ecosystem.py --config results/coevolve-swimming/generation-006.json
```

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
