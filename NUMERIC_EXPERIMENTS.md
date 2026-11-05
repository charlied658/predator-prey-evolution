# Numeric evolutionary experiments

Compare evolutionary algorithms using fitness and measured outcomes, without
watching or rendering the arena. The same ecosystem code provides an optional
visual replay, so there is only one implementation of the simulation rules.

## Run a comparison

The local `.venv` already contains the plotting dependency. From the project root:

```bash
.venv/bin/python evolution/compare_ecosystem.py --output results/my-comparison
```

On another machine, create the environment first (Godot remains required):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Defaults compare **three methods, four candidate controllers, three generations,
one optimizer run, and 10-second episodes**. One fixed scenario is used for
training and a separate fixed scenario for validation. The upper bound is
54 episodes; repeated genomes are cached. Default CPU concurrency is one worker.

No windows, cameras, meshes, or simulation UI are created during headless
experiments. Numeric episodes retain 60 Hz decisions and movement. Several
settings are evaluated in one engine launch to reduce startup overhead.

If you want only numeric output with standard-library Python:

```bash
python3 evolution/compare_ecosystem.py --no-plots --output results/csv-only
```

## Algorithms

| Flag | Mechanism |
| --- | --- |
| `es` | Keep the best controller; generate Gaussian-mutated copies. |
| `ga` | Keep one elite; tournament-select parents, uniformly cross their genes, then mutate. |
| `de` | Differential evolution: construct a trial from three population vectors and cross it with a target; keep successful trials. |
| `random` | Optional non-evolutionary control: draw fresh bounded random candidates and retain the best seen. |

The DE variant holds one elite fixed and proposes trials for the other candidates.
It uses differential scale F=0.8 and binomial crossover CR=0.7, with one forced
trial gene. Tournament size for GA is two. ES and GA use the same mutation scale
(default 15% of each gene's bounded interval). All use the existing three/four
gene controller and fitness formulas; physical traits and the other species stay
fixed. DE requires population size >= 4.

For each optimizer seed, every algorithm receives **exactly the same initial
population**, built from the original rules and mutated copies. Generation 0 is
the best member of that initial population, not just the untouched rules.
Each later generation proposes population-size-minus-one candidates. The proposal
budget per method/run is therefore:

```text
population + generations * (population - 1)
```

Training scenarios remain fixed throughout a comparison, and each candidate is
scored by its mean fitness across those scenarios. Validation is measured for the
training-selected champion but never selects parents, replaces the champion, or
affects mutation. Original-rule reference scores are shown as dashed lines.

Caching makes the number of actual engine episodes smaller than the number of
proposal slots. Actual training/validation episode counts and wall time are
recorded separately; equal proposal budgets do not imply identical CPU time.
The `--max-episodes` cap (default 200) is checked before running simulations.

## Read the results

The output contains:

- `fitness.png` and `fitness.svg`: training and validation fitness vs generation.
- `fitness_budget.png` and `.svg`: training fitness vs candidate slots and wall time.
- `history.csv`: all per-generation fitness values, population mean, gene values,
  captures, food eaten, starvation, episode counts, and elapsed time.
- `summary.json`: final per-method means, validation spread when repeated, actual
  episode counts, and total computation time (excluding plotting).
- `experiment.json`: settings, scenario/optimizer seeds, budgets, engine version,
  objective, gene bounds, and source fingerprints.
- `es/run-000/`, `ga/run-000/`, etc.: saved populations, episode-level JSONL data,
  `best.json`, `validation.json`, generation configs, and per-run histories.
- `baseline.json`: original-rule controller on the first training scenario.

With multiple optimizer runs, colored lines show mean fitness and light lines
show individual runs. Shaded areas show the observed minimum–maximum range;
they are **not confidence intervals**. Flat training curves and worse validation
scores remain visible rather than being filtered out.

The prepared local example is `results/ecosystem-numeric-demo`: three methods,
two paired optimizer seeds, one training and one validation map, three
generations, and 10-second episodes. It ran 97 uncached episodes in about
70 seconds on this Mac. It is a pipeline demonstration, not a conclusive ranking.

## Increase coverage deliberately

```bash
# Include a random-search control and two optimizer runs.
.venv/bin/python evolution/compare_ecosystem.py --algorithms es ga de random --runs 2 --output results/comparison-with-control

# Average over more maps; explicitly allow the larger evaluation budget.
.venv/bin/python evolution/compare_ecosystem.py --runs 3 --train-seeds 42 43 --validation-seeds 10042 10043 --max-episodes 400 --output results/comparison-more-seeds

# Study predator evolution instead.
.venv/bin/python evolution/compare_ecosystem.py --role predator --output results/predator-comparison
```

`--seed` controls the first optimizer seed; runs use consecutive optimizer seeds.
`--train-seeds` and `--validation-seeds` control environment randomness. Training
and validation seeds must be distinct. `--population`, `--generations`,
`--duration`, `--mutation`, and `--workers` adjust effort. Use fresh output folders.

More optimizer runs measure search variability; more scenario seeds measure
performance across different environments. A repeated validation suite supports
exploration, but repeated human tuning against its plots makes it unsuitable as
an untouched final test set. Reserve new seeds before a final scientific study.
Wall-clock plots also depend on machine load, caching, and execution order.

## Plot again without simulating

```bash
.venv/bin/python evolution/plot_ecosystem.py results/my-comparison
```

Plotting only reads saved numeric results. It never launches Godot. Results from
`--no-plots` can be plotted later after installing Matplotlib.

## Visualize only when useful

```bash
python3 evolution/ecosystem.py --config results/my-comparison/es/run-000/best.json
python3 evolution/ecosystem.py --config results/my-comparison/de/run-000/validation.json
python3 evolution/ecosystem.py --config results/my-comparison/ga/run-000/generation-0001.json
```

Each file contains the complete controller and environment settings. `best.json`
uses the first training seed; `validation.json` uses the first validation seed.
When a fitness value averages several maps, one visual replay represents one
map, not the aggregate score. Use the demo's `--seed` override to inspect others.

## Verification

```bash
.venv/bin/python -m unittest discover -s evolution -p 'test_*.py' -v
```

Checks operators/elitism/bounds, paired initialization, equal budgets, caching,
independent scenario suites, serial/parallel agreement, numeric/visual state
agreement, real training and replay, and generated plots.
