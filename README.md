# Predator–Prey Evolution

**Current focus: competitive coevolution.** Both predator and prey evolve in an obstacle arena, with archived opponents, generation matchups, and an interactive history viewer. See [COEVOLUTION.md](COEVOLUTION.md) for the prepared demo, training commands, and methods.

```powershell
python evolution/coevolution.py replay results/coevolution-demo
```

Python uses only the standard library. The fixed-prey versus diverse-prey experiments below remain available as earlier baselines.

## Setup

Requires Godot 4.x and Python 3.10+. Open `project.godot` once in Godot to import the project. From the repository root:

```powershell
Copy-Item evolution/config.json.example evolution/config.json
```

Edit the local JSON with absolute paths to the Godot executable and the directory containing `project.godot`. This configuration is ignored by Git. Commands below assume `python` points to a working Python installation; use `py` or the full executable path if needed.

## Start with a small experiment

Run all commands from the repository root:

```powershell
python evolution/experiment.py --output results/pursuit-quick --runs 2 --population 8 --generations 4 --episodes 4 --validation-per-kind 2 --test-per-kind 4 --workers 2
```

For the default study (10 paired runs, 30 individuals, 30 generations, 8 training encounters per individual):

```powershell
python evolution/experiment.py --output results/pursuit-study --workers 2
```

Benchmark a small run before committing to the full study. The default study runs 167,200 encounters including validation and final testing; hardware and process startup costs affect runtime. Workers split each batch across separate headless Godot processes and preserve input order. Each process uses isolated temporary files. Avoid launching more workers than your machine can handle comfortably.

Resume an interrupted study using the **same command and settings**, adding `--resume`. A checkpoint is saved after each completed generation. A partially completed generation is repeated. Source hashes and experiment settings must match; use a new output folder after changing code or settings. Do not run two invocations into the same output directory.

## Training modes and the pursuit fix

The default `--training-mode pursuit` fixes a weak behavioral baseline found in the original study. Both comparison conditions use the same revised profile:

- Predator position inputs are the unit bearing to the prey, rather than small displacements divided by 30. Opponent velocity and distance inputs remain available. Prey sensing remains unchanged.
- Initial hidden/output biases and unsigned-distance input weights are zero, avoiding arbitrary initial movement preferences. All 66 parameters remain evolvable; no chase rule is hardcoded into the predator.
- The first third of generations trains against stationary targets from randomized positions. Later generations retain stationary targets in even-numbered episode slots, with fixed/pool neural opponents in the remaining slots.
- Unsuccessful encounters receive `10 * max(-1, 1 - mean_distance / initial_distance) + 5 * approach_alignment`. Alignment measures actual predator motion toward the prey, normalized by its speed and time. Prey approaching the predator and pushing against walls do not earn alignment credit. Captures retain the original reward.
- Reproduction uses mutation and elitism, avoiding uniform crossover that can disrupt coordinated neural weights.
- Validation capture rate selects the champion, with mean pursuit fitness breaking ties.

Use `--training-mode legacy` for the original initialization, displacement inputs, pure fixed/pool training, and uniform crossover. Legacy failures receive zero, or the old minimum-distance reward with `--shaping`. In pursuit mode, dense feedback is always enabled and `--shaping` has no additional effect.

These are combined algorithm/representation changes, not an isolated fitness ablation. Compare fixed and diverse **within the same mode**. Old study outputs remain readable and replayable; changed source hashes prevent resuming those studies with the revised code. Use a new output folder.

Only predators evolve. Neural prey are fixed random networks, and scripted prey do not learn. Straight movement by prey is therefore expected.

The exploratory `results/pursuit-bearing-check` run (seed 100, 30 individuals, 15 generations) reached 80/80 test captures for each condition, including 20/20 stationary targets. The initial winner already scored 84% on validation, so the sensing/initialization prior contributes substantially. This confirms basic pursuit but makes the current benchmark too easy to resolve the scientific comparison. It is not a fresh confirmatory study: the evaluation design has been consulted during development. Before the next final study, introduce a harder controlled task and reserve new evaluation seeds.

## What the experiment compares

**Question:** Does training against a diverse pool of prey improve generalization compared with training against one fixed prey?

- `fixed`: one randomly generated neural prey controller per training seed.
- `diverse`: a pool of `--pool-size` neural prey controllers (default 8), sampled independently for each encounter. The fixed prey is the first member of that same pool.
- Both methods receive identical initial predator populations, starting-position schedules, number of training encounters, and validation budgets for each seed.
- Starts are sampled inside [-12, 12] on both axes, with initial separation between 5 and 20. Movement is clamped at arena coordinates ±14.
- Every candidate in a generation faces the same cases. New training cases are sampled each generation.
- Both agents decide using the same previous-step state, then move simultaneously at a fixed 60 Hz. Speed limits are equal (3 units/second); capture radius is 1 and timeout is 20 seconds.
- Neural controllers have 5 inputs, 8 hidden neurons, 2 outputs, and 66 evolved weights/biases. There are no headings; movement outputs are world-axis velocities.
- Both modes use tournament size 3, per-weight Gaussian mutation probability 0.1 and standard deviation 0.2, and 2 elites. Only legacy mode uses uniform crossover.

Training fitness is the mean encounter reward. Captures receive `100 + (20 - capture_time)`. Failure rewards depend on the training mode described above. Capture dominates within an encounter, but averaged fitness can trade capture frequency against capture speed.

Each generation's training winner is evaluated on a fixed validation suite. The saved champion is the candidate with the highest validation capture rate; pursuit-mode ties use mean pursuit fitness, while legacy ties keep the earlier candidate. Validation therefore participates in model selection and is **not** an unbiased test. The final test suite is used only after training. Validation and test suites have distinct fixed seeds and include equal numbers of:

1. New random neural controllers (a separate controller per case).
2. Stationary prey.
3. Prey moving along the positive z axis.
4. Prey fleeing directly away from the predator.

Straight and flee prey are evaluation-only. Stationary prey are also used for pursuit-mode training. Straight prey eventually stop against the boundary; direct-flee prey can become trapped. They are reference behaviors, not optimal adversaries. Shared test cases make comparisons paired, and the saved suite files make the evaluation inspectable. Do not tune repeatedly against test results; create a new untouched final suite if using these results for exploratory development.

## Outputs and interpretation

Each study folder contains:

- `experiment.json`, `source_hashes.json`: settings and implementation fingerprints.
- `validation_suite.json`, `test_suite.json`: explicit evaluation scenarios.
- `runs.csv`: one row per independently trained controller, including capture rate, capture-only duration, minimum distance, path length, and boundary-time fractions.
- `learning_curves.csv`: training fitness and validation capture rates against generation and encounter budget.
- `summary.json`: group means and paired diverse-minus-fixed difference, with deterministic percentile-bootstrap 95% intervals across training seeds (5,000 resamples). A single run has no interval.
- `capture_rates.svg`: individual final test capture rates, means, and bootstrap intervals; open in a browser.
- `timing.json`: wall time for the most recent invocation, including final evaluation and reporting. Resumed invocations do not represent complete-study runtime.
- Per-condition/seed folders: `checkpoint.json`, `best_predator.json`, `report.json` (including opponent-specific metrics), and `test_encounters.json` containing every test result and scenario.

Time to capture excludes timeouts; absent captures are represented by null/blank. Boundary time means the agent is within 0.1 units of an arena edge. Path length measures actual movement after clamping. Minimum distance includes the starting distance.

The primary endpoint is final held-out capture rate. Inspect opponent-specific results so an aggregate gain does not hide a regression against fleeing prey. Confidence intervals resample independent training runs, not individual encounters; they describe training-seed variability conditional on this particular test suite. A small pilot checks the workflow and is not sufficient evidence of superiority. About 10 or more runs is a starting point, not a guarantee of statistical power.

The learning curves use validation, not test performance, to avoid repeatedly consulting the held-out test suite. Both methods have equal training and validation costs; the CSV records both training-only and total selection encounter counts. Default per condition/seed: 7,200 training + 960 validation + 200 final test encounters.

## Replay and validation

Replay requires an explicit controller JSON, run directory, or study directory. It prints the exact controller, seed, prey type, and recorded outcome. When given a study directory, it selects the requested condition/seed; without a seed, it chooses the best validation champion, never the best test score. For the corrected controller already generated:

```powershell
python evolution/replay.py results/pursuit-bearing-check --condition diverse --seed 100 --opponent flee
python evolution/replay.py results/pursuit-bearing-check --condition diverse --seed 100 --opponent stationary --case-index 3
```

Run unit and actual Godot integration checks:

```powershell
python -m unittest discover -s evolution -p 'test_*.py' -v
```

Eight checks cover seeded scenarios, initialization, explicit replay selection, reward shaping, known stationary/pursuit trajectories, boundary metrics, identical serial/parallel results, malformed genomes, and exact interrupted/resumed results in both training modes.

The original single-encounter training script remains available:

```powershell
python evolution/genetic_algorithm.py
python evolution/replay.py best_predator.json
```

Its historical results used sequential agent updates; retrain rather than directly comparing those old scores to this experiment.

## Next extensions

Once fixed-versus-diverse training has been evaluated, useful follow-ups include coevolution with archived opponents, wall-distance sensors, or testing unseen speed ratios. Introduce one factor at a time and hold evaluation budgets constant. Current limitations include easy/erratic random opponents, no explicit wall sensing, equal speeds, a single arena geometry, and no independent sampling of multiple final test suites.
