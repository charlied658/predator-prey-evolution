# Competitive coevolution

The current project evolves **both** predator and prey neural controllers. The aim is to inspect changing pursuit/evasion strategies, historical improvement, and cycles of counterstrategies. Complex behavior is an outcome to investigate, not a guaranteed result.

## Watch the prepared demo

From the repository root, using your configured Python and Godot:

```powershell
python evolution/coevolution.py replay results/coevolution-demo
```

The playlist compares initial-vs-initial, final-predator-vs-initial-prey, initial-predator-vs-final-prey, and final-vs-final on the same three scenarios. Red is predator; green is prey. The overlay identifies each generation. **Space** changes matchup, **R** restarts, and **Esc** exits. Encounters pause at capture/timeout so you can inspect trails.

For a particular historical matchup:

```powershell
python evolution/coevolution.py replay results/coevolution-demo --predator-generation 5 --prey-generation 20 --scenarios 4
```

Open `results/coevolution-demo/dashboard.html` in a browser. It is a standalone local report requiring no server or external dependencies. Select a generation and scrub time to inspect paths. The historical matrix shows each predator generation's capture rate against each prey generation. Every cell uses the same starting scenarios.

## Train

A small run:

```powershell
python evolution/coevolution.py train --output results/coevolution-small --population 16 --generations 10 --episodes 6 --reference-cases 4 --matrix-every 5 --matrix-cases 6 --workers 2
```

A longer run with the defaults (40 individuals per species, 60 generations):

```powershell
python evolution/coevolution.py train --output results/coevolution-long --seed 200 --workers 2
```

Use `--arena open` for an obstacle-free control. Different seeds and arenas belong in different output folders. For stronger evidence, repeat independent runs with several seeds; do not assume one run's visually interesting trajectory represents all runs.

Resume with the original command and settings plus `--resume`. Settings, Python version, and source hashes must match. Both populations, archives, random state, and history are checkpointed after each generation. An interrupted generation is repeated deterministically. Analysis can be rerun separately without retraining:

```powershell
python evolution/coevolution.py analyze results/coevolution-demo --matrix-every 2 --matrix-cases 20 --workers 2
```

A finer matrix costs more encounters: approximately the square of the number of selected generations times `matrix-cases`. This replaces derived analysis/replay files; it does not alter the training checkpoints. Training and analysis should not run concurrently into the same output folder.

## Simulation

Both species have the same 15-input, 12-hidden-unit, 2-output tanh network: **218 evolvable weights and biases**. Inputs are opponent bearing (2), opponent actual velocity (2), distance (1), own actual velocity (2), and eight equally spaced radial clearance sensors (8). Output is planar movement, capped at 3 units/second. Neural controllers are feedforward: they have velocity feedback but no recurrent hidden state. There is no scripted chase/flee action in coevolution training.

Starts are randomized with separation 5–20, and do not overlap obstacles. Encounters last at most 20 simulated seconds at 60 Hz; capture distance is 1. Both agents decide from the same state, then move. In ecology mode, velocity observations reflect actual movement after contacts, rather than attempted movement into a wall.

The obstacle arena has three circular barriers of radius 2. Agents have radius 0.5 for obstacle clearance. The displayed cylinders correspond to these analytic 2D collision boundaries. Swept-circle intersection and tangential sliding prevent tunneling through obstacles. Agent centers are bounded at ±14 on each axis. Capture is blocked when a barrier separates the agents. Eight range sensors report clearance over 6 units, including walls and inflated obstacle boundaries. Obstacles do **not** hide the opponent's position; these are movement barriers, not visual refuges.

The original 66-weight controllers and earlier replay files remain supported. They use their original sensor defaults; they are not directly interchangeable with the expanded coevolution genomes.

## Selection and archives

Each species has a separate population. In each generation, every candidate of a species faces the same opponent/scenario schedule. Approximately half of the episodes sample the current opposing population, and half sample archived opposing champions (uniformly, with replacement). Both species are evaluated against the populations from before reproduction. Archives are updated only after both evaluations, avoiding an asymmetric update advantage.

Each generation saves the training winner for each species. A champion means best on that generation's sampled opponents, not globally best. Generation 0 is an unselected random baseline, not an evolved champion. Archives retain every generation, including generation 0. With odd episode counts, the extra episode uses a current opponent.

Predator reward is `100 + (20 - capture_time)` on capture. Failed encounters get sustained-closeness and approach feedback: `10 * max(-1, 1 - mean_distance / initial_distance) + 5 * approach_alignment`. Alignment counts the predator's own actual progress toward prey, so prey moving toward a stationary predator and wall pushing do not earn approach credit.

Prey reward is survival seconds plus a small bounded separation term: `time + 0.5 * min(mean_distance / 30, 1)`. Thus survival dominates broadly, but differences below half a second can be traded against separation. Neither species earns rewards for turning or obstacle proximity; these are measurements, not prescribed tactics.

Fitness is averaged over encounters. Reproduction uses tournament selection (size 3), two elites, and Gaussian mutation (probability 0.1 per parameter, standard deviation 0.2). It does not use crossover. Initial neural biases and unsigned-distance weights are zero; sensor/own-velocity weights start small. All parameters can subsequently mutate.

## Read the results

- `generations/generation-NNNN.json`: both champions and metrics for every generation, including generation 0.
- `checkpoint.json`: next populations, opponent archives, RNG state, and history.
- `config.json`, `provenance.json`: experiment settings, Python version, and source fingerprints.
- `history.csv`: fitness, champion training scores, fixed-reference scores, and same-generation behavior metrics.
- `historical_tournament.csv` / `.json`: sampled predator/prey generation cross-play. JSON includes every starting scenario.
- `trajectories.json`: one fixed-scenario encounter per generation, with sampled positions every 0.2 seconds plus the end state.
- `dashboard.html`: local interactive trajectory viewer and historical matrix.
- `replay.json`: Godot comparison playlist.

The fixed references evaluate each new predator against the initial prey, each new prey against the initial predator, and the current champions against each other. These measurements do not select champions or influence reproduction. Tournament scenarios use a separate seed from training/reference scenarios. They are descriptive historical diagnostics, not a repeatedly untouched statistical test.

Path length measures actual movement. Turn angle is cumulative absolute change of movement direction, in radians; the dashboard divides by 2π for readability. It includes collision sliding and oscillation, so it is **not** a count of deliberate circling maneuvers. Blocked time counts near-zero motion while attempting movement. Obstacle time counts time within one unit of an inflated obstacle boundary. Traces are sampled and can visually cut corners between samples; collision tests use the underlying simulation steps.

A useful investigation compares early and late opponents across several scenarios, then checks whether the pattern repeats across seeds. Stronger historical performance supports improvement; alternating strengths can suggest strategy cycles. A path near cover or a large turning score alone does not establish an intelligent tactic. Current limitations include equal fixed speeds, omniscient opponent sensing, no energy budget, no group behavior, one obstacle layout, and no recurrent memory.

## Verify

```powershell
python -m unittest discover -s evolution -p 'test_*.py' -v
```

Tests cover both populations changing, current/archive sampling, safe starts, collision blocking/sliding, sensor response, trace safety, serial/parallel consistency, and exact interrupted/resumed coevolution, plus the earlier experiment checks.
