# Ecosystem lab

For numeric-first algorithm comparisons and fitness plots, start with [NUMERIC_EXPERIMENTS.md](NUMERIC_EXPERIMENTS.md). The simulation view below is optional.

A multi-agent ecosystem with explicit hunting, evasion, foraging, and avoidance
rules. Basic evolution tunes these rules through named numeric genes. Movement
uses swept-circle rock collisions and sliding in `scripts/agent.gd`.

## Watch a small demo

From the repository root (Python 3.10+ and Godot 4.7, with your local
`evolution/config.json` set up as in README):

```bash
python3 evolution/ecosystem.py
```

Or open `ecosystem.tscn` in Godot and press **F6** (Run Current Scene).
The default is 2 predators, 12 prey, 14 rocks, 28 food pellets, and 90 simulated
seconds. Red animals hunt; green animals evade and forage. White dots show
heading; yellow bars show remaining energy. Colored floor discs are home bases.

- **Space:** pause/resume.
- **R:** restart the same seed.
- **N:** restart with the next seed and a newly generated environment.
- **1 / 2 / 4:** simulation speed.
- **Esc:** quit.

The final scene stays visible after the timer finishes. R/N start another round.
Settings are loaded on launch; restart the command after editing the JSON.
Pressing F5 also opens the ecosystem, the project's default scene.

## Try ideas quickly

Edit `evolution/ecosystem.json`, or copy it for a separate scenario:

```bash
python3 evolution/ecosystem.py --predators 1 --prey 20 --seed 7 --duration 60
python3 evolution/ecosystem.py --config evolution/ecosystem.json --headless --duration 30 --output results/ecosystem-seed42.json
python3 evolution/ecosystem.py --headless --seed 43 --duration 30 --output results/ecosystem-seed43.json
```

Each invocation saves a Godot `.log` beside the output and the exact settings to `<output>.config.json`. On normal
completion it writes a summary containing captures, food consumption, starvation
counts by species, spawns, and mean energy while alive. The default output is
`results/ecosystem-summary.json`; use distinct output paths to retain comparisons.
Closing early with Esc does not write a completed summary. Same settings and seed
are repeatable within the same Godot version; cross-version or cross-platform
floating-point equality is not guaranteed.

## Mechanics and knobs

| Settings | Meaning |
| --- | --- |
| `predators`, `prey` | Population slots, replenished by respawning; no reproduction yet. |
| `rocks`, `seed` | Seeded, non-overlapping circular rocks with corridors and clear bases. Impossible layouts report an error. |
| `food_pellets`, `food_regrowth` | Prey-only food; consumed pellets regrow at a new open location after this many seconds. |
| `food_energy`, `capture_energy` | Energy restored by eating food or catching prey; capped at 100. |
| `*_hunger_drain`, `movement_hunger_drain` | Energy lost per simulated second, plus a cost proportional to actual movement speed. |
| `hunger_speed_floor` | Speed at zero energy as a fraction of full speed. Speed drops linearly with energy. |
| `*_speed`, `acceleration` | Full-energy speed (arena units/second), acceleration (units/second²). |
| `turn_degrees_per_second` | Maximum heading turn rate. Velocity changes gradually, producing inertia. |
| `*_vision`, `*_fov` | Visual range and cone width in degrees around the white heading dot. Rocks occlude targets and food. |
| `*_home`, `home_radius` | `[x,z]` spawn centers and radius, kept free of rocks. Bases do not replenish energy or grant permanent safety. |
| `respawn_delay`, `spawn_grace` | Delay after death; capture immunity for either animal immediately after spawning. Hunger still drains. |
| `duration` | Round length in simulated seconds. |

Predators chase the nearest visible prey, otherwise wander. Prey flee the nearest
visible predator; if none is visible and energy is below 85, they seek visible
food. Otherwise they wander. Both steer away from nearby walls, rocks, and
animals. Local avoidance acts like touch/proximity sensing and works outside the
visual cone. Food is eaten at contact below 95 energy, and capture requires bodies
to come within 1 unit with an unobstructed segment. Multiple predators can chase
the same prey, but a prey can only be captured once per step.

All decisions are computed before animals move. Capture resolves before feeding
and starvation; an animal can be saved from starvation by a meal that step.
Dead animals disappear and respawn at their own base with full energy. Soft
separation reduces crowding; animals do not have hard physical collisions with
one another. This is simplified planar locomotion, not a biological model.

Use short demos to investigate resource scarcity, population ratios, vision,
and terrain. For parameter evolution and its genetic representation, see below
and the [genome explanation in README](README.md#what-is-the-genome).

## Focused mechanic check

```bash
/Applications/Godot.app/Contents/MacOS/Godot --headless --log-file /tmp/ecosystem-check.log --path . --script tests/ecosystem_smoke.gd
```

Checks seeded reset, populations, vision range/cone, rock occlusion, capture,
energy replenishment, starvation, home respawning, and food regrowth. Use your
Godot executable path on other platforms.

## Basic evolutionary algorithm

Run a small prey experiment:

```bash
python3 evolution/evolve_ecosystem.py
```

This keeps the working scripted behavior and evolves **four numbers**: evasion
weight (`target_weight`), obstacle avoidance, wandering, and the energy threshold
for seeking food. All prey in an episode share one candidate's parameters.
Predators keep the original behavior. Use `--role predator` to evolve three
predator parameters instead (pursuit, avoidance, and wandering), against the
original prey. The ecosystem, physical traits, and movement rules stay fixed.

The algorithm is a mutation-only evolution strategy:

```text
winner = original behavior
repeat for each generation:
    candidates = winner + small randomly mutated copies
    run each candidate on the same map/seed
    winner = candidate with the highest fitness
```

The unchanged winner survives each generation. Ties keep the existing winner.
There are no neural networks, crossover, or simultaneous coevolution in this
minimal version. A new candidate population is evaluated between generations;
individual animals do not learn during an episode or mutate when respawning.

Default budget: **4 candidate controllers, 3 generations, 15 seconds per episode**.
The fixed seed lets the unchanged winner's result be reused, so this needs at
most **10 simulated episodes**. Candidate population size is separate from the
2 predators / 12 prey in the arena.

Fitness is intentionally simple:

```text
predator: captures - 2 * predator starvations
prey:     food eaten - 2 * captures - 2 * prey starvations
```

These scores allow tradeoffs: two food pellets offset one capture penalty.
Short runs may show no starvation. Scores can improve on the chosen map without
improving on others; this is a quick experiment, not a generalization study.

A prepared local run is already saved in `results/ecosystem-evolution-basic`
(original fitness 4, final fitness 8 on seed 42). To train another run, specify a
fresh `--output` folder. Watch the original and evolved behavior with the existing demo command:

```bash
python3 evolution/ecosystem.py --config results/ecosystem-evolution-basic/baseline.json
python3 evolution/ecosystem.py --config results/ecosystem-evolution-basic/best.json
python3 evolution/ecosystem.py --config results/ecosystem-evolution-basic/generation-0001.json
```

Try a new seed or evolve predators in a separate output folder:

```bash
python3 evolution/evolve_ecosystem.py --seed 7 --output results/evolution-seed7
python3 evolution/evolve_ecosystem.py --role predator --output results/evolution-predator
```

`--population`, `--generations`, `--duration`, and `--mutation` adjust the small
experiment. Mutation defaults to Gaussian noise with standard deviation 15% of
each gene's allowed interval, clamped to that interval. Use a new output folder
for each run; this minimal implementation does not resume interrupted training.

Each run saves `baseline.json`, `best.json`, per-generation winner configs,
`population-*.json` with every candidate's genes and fitness, `history.csv`, the
fixed experiment settings, individual episode configs/results/logs, and timing.
Generation 0 is the original behavior; generations 1 onward select among it and
its mutated descendants. Saved generation configs run directly in the ecosystem.
An optional `behavior` object holds the species' parameters; omitting it preserves
the original controller defaults.

Focused tests:

```bash
python3 -m unittest discover -s evolution -p test_evolve_ecosystem.py -v
```
