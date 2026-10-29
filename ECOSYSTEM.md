# Ecosystem lab

A quick sandbox for exploring multiple animals, resources, and sensing. This is a
**rule-based ecology demo**, not a new evolved policy or a replacement for the
existing pairwise neural coevolution experiment. It reuses the original agent's
swept-circle rock collisions and sliding. Original training/replay commands and
fitness functions are unchanged.

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
Pressing F5 still opens the original pairwise scene.

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

For now, use these short demos to investigate resource scarcity, population
ratios, vision, and terrain. The existing 66/218-weight genomes have no food,
hunger, or multi-agent input contract, so they are deliberately not silently
reused here. Extending that neural sensor contract and adding ecosystem fitness
is a separate experimental step once the mechanics are settled.

## Focused mechanic check

```bash
/Applications/Godot.app/Contents/MacOS/Godot --headless --log-file /tmp/ecosystem-check.log --path . --script tests/ecosystem_smoke.gd
```

Checks seeded reset, populations, vision range/cone, rock occlusion, capture,
energy replenishment, starvation, home respawning, and food regrowth. Use your
Godot executable path on other platforms.
