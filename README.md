# Predatorâ€“Prey Evolution

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

## Vision-range pilot on procedural terrain

This isolated NumPy pilot uses the exported broad-outcrops map and the swept
circle collision solver. It compares equal vision (2 world units each), longer
prey vision (2 versus 4), and longer predator vision (4 versus 2). Only neural
network weights evolve. Speeds are equal, prey collect 12 finite pellets, and obstacles do not
block sight. This does not change the main Godot ecosystem.

```powershell
.\.venv\Scripts\python.exe evolution/vision_pilot.py --output results/vision-pilot-next
```

Requires NumPy, Matplotlib, Pillow, and the map exported to
`results/landscape-previews/map-0-seed-42.json` by the landscape preview.
Defaults: three seeds, 12 generations, four individuals per role. Each generation
uses all population pairings on two training starts; evaluation uses 12 separate
starts. Outputs include measurements, genomes, terrain, a comparison figure,
and two animated replays. Range circles show sensing distance; white body
outlines indicate that the organism currently sees its opponent. Replays freeze
when capture occurs. This small pilot tests feasibility, not statistical certainty.

The food version adds 0.25 prey reward per pellet alongside survival, with no
hunger. Prey sense the nearest uncollected pellet within their vision range;
longer vision therefore helps detect both food and opponents. Replays show
remaining food and collection totals, and the results figure includes collection.

Training arrangements now change every generation by default (`--training-mode
resampled`). All candidate pairings share the same two arrangements within a
generation. Starting positions and pellet layouts change together; the 12
held-out evaluation arrangements remain fixed. Use `--training-mode fixed` to
reproduce the earlier experiment with two repeated training arrangements.

For an isolated population-size check, run:

```powershell
.\.venv\Scripts\python.exe -u evolution/vision_pilot.py --foraging-population-test --generations 50 --seeds 3 --eval-every 5 --output results/foraging-population-next
```

This compares 4 and 16 prey controllers without capture or opponent inputs.
Food sensing, movement, and pellet rewards are unchanged; no approach reward is
added. Training arrangements vary; each seed has 24 fixed held-out encounters.
Population 16 uses four times the training evaluations at equal generations.
The output includes a comparison figure and a final food-sensor ablation check.

To test food representation and obstacles independently:

```powershell
.\.venv\Scripts\python.exe -u evolution/vision_pilot.py --foraging-factor-test --generations 50 --seeds 3 --eval-every 5 --output results/foraging-factors-next
```

This food-only 2-by-2 diagnostic compares original food displacement inputs
with unit direction inputs, in the terrain map and an empty arena. Both arenas
use identical starts and pellet locations drawn from terrain-valid sites.
Population remains 16, vision remains 24 cells, and only pellet collection earns
variable reward. No approach reward or hunger is added. Unit direction removes
the distance information previously implicit in displacement magnitude, without
changing the network size. The normal vision experiment retains original inputs.

Restore predator/prey co-evolution in an empty arena with improved food inputs:

```powershell
.\.venv\Scripts\python.exe -u evolution/vision_pilot.py --arena empty --food-encoding unit --generations 50 --seeds 3 --eval-every 5 --output results/empty-predator-50
```

This starts fresh with four controllers per role and runs the three vision
conditions. Capture ends the encounter. Speeds are equal, prey receive survival
and pellet rewards, and training arrangements change each generation. Graphs,
measurements, and vision-circle replays are saved in the output folder.

Normal pursuit runs now save `history.json` at each evaluation checkpoint.
After a run with at least two seeds, compare contemporary, fixed-panel, and
historical opponents with:

```powershell
.\.venv\Scripts\python.exe evolution/evaluate_pursuit_progress.py results/empty-predator-progress-50
```

Outputs go into `progress-diagnostic` inside that run. The stable panel contains
initial and final controllers from the other independent evolutionary seeds.
This is retrospective evaluation only and never affects training. The diagnostic
also measures food collection without capture or opponent inputs and produces
historical capture matrices. Existing runs without archived controllers need to
be reproduced before these comparisons are possible.

Selection reliability can be tested by changing `--training-starts` (1 to 8,
default 2). Use `--conditions equal` to run only the equal-vision condition.
For example, `--training-starts 8 --conditions equal` gives every candidate the
same eight fresh arrangements per generation. This includes the original two
arrangements and uses four times their training evaluations.

The paired selection diagnostic compares an eight-encounter run with a saved
two-encounter baseline using a single shared fixed panel and fresh test starts:

```powershell
.\.venv\Scripts\python.exe evolution/evaluate_pursuit_progress.py results/selection-eight-50 --compare-training results/empty-predator-progress-50
```

Run fixed-opponent evolution versus co-evolution, including shared benchmarks:

```powershell
.\.venv\Scripts\python.exe -u evolution/run_opponent_comparison.py --generations 50 --seeds 3 --output results/opponent-comparison-50
```

Both treatments use equal vision, the empty arena, improved food inputs, eight
fresh starts per generation, identical initial controllers and mutation streams.
The fixed treatment evolves predator and prey separately against frozen initial
opponents. Both treatments use the same number of generations and candidate
controllers. Deterministic repeat matches against a fixed opponent are omitted;
actual simulation costs therefore differ. This is a matched search-opportunity
comparison, not a strict equal-compute experiment. A shared retrospective panel
from the other co-evolution seeds tests both treatments. All stages print progress.
Results and replays are grouped under one output directory; the comparison figure
is `fixed/opponent-comparison/opponent-comparison.png`.

For the 10-seed replication with mean curves and confidence bands:

```powershell
.\.venv\Scripts\python.exe -u evolution/run_opponent_comparison.py --generations 50 --seeds 10 --seed-offset 10 --benchmark-reference results/opponent-comparison-50 --output results/opponent-comparison-10seeds
```

The frozen benchmark contains final controllers from BOTH treatments of the
previous three-seed pilot. The new run uses separate evolutionary seeds 10–19.
Plots show means and pointwise 95% percentile bootstrap confidence intervals,
resampling complete seed trajectories 10,000 times. Games within a seed are
averaged before bootstrap; they are not counted as independent replicates.
Bands describe seed variation conditional on this frozen panel, not uncertainty
across all possible benchmark opponents. `paired-differences.png` bootstraps
matched seed pairs; positive differences favour co-evolution. The intervals are
pointwise, not simultaneous tests across all checkpoints. The primary comparison
still matches candidate generations and starts rather than total simulator cost.

To probe whether prey approach or avoid sensed predators without training:

```powershell
.\.venv\Scripts\python.exe -u evolution/probe_prey_sensing.py results/opponent-comparison-10seeds --reference results/opponent-comparison-50
```

This holds food and body state fixed while varying predator bearings and range.
It compares immediate movements with prey opponent inputs enabled versus masked,
then reruns the existing six-predator benchmark with those inputs masked. Physical
capture remains active. Results go to `prey-sensing-check`; the normal experiment
is unchanged. Negative radial change means sensing steers away from the predator;
positive benchmark differences mean sensing improves survival or collection.

Minimal one-prey/one-pellet diagnostic (no predator or terrain):

```powershell
.\.venv\Scripts\python.exe -u evolution/single_pellet.py --generations 50 --seeds 10 --output results/single-pellet-next
```

Uses the same network, unit food input, body radius, movement and collection
rules. Eight fresh arrangements per generation; 128 held-out tests per seed.
The only evolutionary reward is reaching the pellet. A direct-steering reference
verifies the task is solvable and is never inserted into evolution. Outputs
include success-rate confidence bands, per-seed results and a replay of the first
seed's final controller on the first three test arrangements. No settings in the
predator/prey experiments are changed.

The same minimal diagnostic supports multiple pellets without changing the
controller or adding predator pressure:

```powershell
.\.venv\Scripts\python.exe -u evolution/single_pellet.py --pellet-counts 4 12 --baseline results/single-pellet-50 --output results/pellet-count-next
```

Larger pellet sets contain the smaller sets at identical starts. Food is initially
within vision, spaced at least six cells apart. Fitness counts collected pellets;
the episode continues until all are collected or ten seconds elapse. The root
comparison plots absolute count, fraction collected and all-food completion with
seed-based confidence intervals. Each condition has its own replay and data
subfolder. Direct steering remains limited to visible food, so it may stop when
remaining pellets are out of range; it is not an optimal routing reference.

Compare whole-map scattered food with and without the complex landscape:

```powershell
.\.venv\Scripts\python.exe -u evolution/single_pellet.py --compare-landscapes --pellet-counts 12 --generations 50 --seeds 10 --output results/scattered-landscape-50
```

This starts both conditions from scratch, without predators. The same starts and
food layouts are drawn from the complex map's largest body-accessible region,
which spans the arena. The empty control uses these same sites. All pellets are
reachable, but may begin outside vision. Vision remains 24 cells with no obstacle
occlusion. Speed, ten-second horizon, rewards, network and evolutionary settings
remain unchanged. Root outputs contain mean/95% CI comparison plots; `empty` and
`terrain` subfolders contain data and actual-terrain replays. The direct reference
still only steers toward visible food; it has no pathfinding or exploration rule.

Test an exploration reward against food-only fitness in both landscapes:

```powershell
.\.venv\Scripts\python.exe -u evolution/single_pellet.py --compare-exploration --pellet-counts 12 --generations 50 --seeds 10 --exploration-reward 0.1 --grid-cells 8 --output results/exploration-reward-50
```

Fitness is pellets collected plus 0.1 per new centre-occupied tile after the
starting tile. The 8-by-8 grid resets each encounter; revisits earn nothing.
The organism does NOT receive this grid as an input. Replays overlay grid lines:
green open areas are visited, cream areas unvisited, and gray terrain stays gray.
A tile is visited when the body's centre occupies it at a simulation timestep,
not when it becomes visible. The root figure compares FOOD collection and tile
coverage separately, with seed-based confidence bands. Results are grouped by
`empty`/`terrain`, then `food-only`/`exploration`, with replays inside `12-pellets`.
The full four-condition run preserves other learning settings and prints progress.

Compare feedforward and recurrent controllers without predators:

```powershell
.\.venv\Scripts\python.exe -u evolution/single_pellet.py --compare-memory --pellet-counts 12 --generations 50 --seeds 10 --exploration-reward 0 --output results/controller-memory-50
```

Uses scattered food in matched empty/complex arenas. Feedforward controllers have
8 hidden units and 170 weights. Recurrent controllers have 6 hidden units and
164 weights, including a 6-by-6 recurrent matrix. Both see the same 18 inputs;
the visited grid remains evaluator-only. Hidden state starts at zero for EACH
encounter. A third curve evaluates each recurrent champion with state reset every
step, without training a new controller or feeding ablation scores to selection.
Both training treatments use food-only reward, 4 candidates, 8 training starts,
128 fixed test starts and identical mutation settings. Matching seeds pair layouts
and RNG seeds, not initial behaviours, because controller shapes differ.
Results contain mean/95% CI curves, paired endpoint differences and grid-overlay
replays grouped under `empty`/`terrain`, then `feedforward`/`recurrent`. A reset
penalty establishes use of temporal state, not necessarily sophisticated memory.

Test an explicit visited-area map in the EMPTY arena only:

```powershell
.\.venv\Scripts\python.exe -u evolution/single_pellet.py --compare-visited-map --pellet-counts 12 --generations 50 --seeds 10 --exploration-reward 0 --grid-cells 8 --output results/visited-map-50
```

Both trained controllers receive local observations and absolute x/y position.
The map treatment also receives 64 binary centre-visit flags; the position-only
control has these channels fixed to zero. Both networks have 84 inputs, 8 hidden
units and 698 weights. The same map-trained controller is additionally evaluated
with visit history blanked while keeping position and local sensors unchanged.
Map flags reset each encounter and never reveal food outside vision. Replays
retain the visited-grid overlay. Food is scattered across the entire empty arena;
there is no terrain, predator, or exploration reward. Mutation retains the old
per-weight rate (49 changed weights per child for this larger genome, sigma .35).
Thus this tests history input with matched architectures, rather than comparing
the larger network directly with the old 170-weight baseline. Outputs contain
mean/95% CI food and coverage curves plus paired endpoint differences.

Completed foraging conditions are now automatically reusable. Each new
`run_count` saves a compatibility fingerprint covering settings, controller,
world arrays, simulation/training code, and artifact hashes. A registry under
`results/.foraging-cache` locates exact matches. Matching runs print `REUSED` and
copy saved results/replays into the new experiment; they do not retrain. Missing,
incomplete, incompatible or modified automatic cache entries are rejected.
Completed sources are never modified. Keep source run folders to retain reuse.

For the visited-map experiment, an existing position-only control can also be
selected explicitly:

```powershell
.\.venv\Scripts\python.exe -u evolution/single_pellet.py --compare-visited-map --pellet-counts 12 --generations 50 --seeds 10 --exploration-reward 0 --grid-cells 8 --reuse-control results/visited-map-50/position-only/12-pellets --output results/visited-map-followup
```

The source must be complete and match exactly. Older runs without a fingerprint
are checked against their saved configuration and their initial/final held-out
scores are re-evaluated, without evolution. This validates numerical compatibility
but cannot reconstruct missing training provenance. Verified copies receive a
fingerprint for future reuse. A 170-weight feedforward run is NOT interchangeable
with the 698-weight position-only control. For a first visit-map experiment this
new control must still be trained once. Unlike a fresh search, legacy verification
is a small fixed evaluation cost. Broader historical benchmarks can still be
compared descriptively, without treating different configurations as controls.

Explicit forage/explore instructions (empty arena only):

```powershell
.\.venv\Scripts\python.exe -u evolution/single_pellet.py --hybrid-test --pellet-counts 12 --generations 20 --seeds 5 --output results/hybrid-instructions-next
```

The authoritative rules are in `evolution/hybrid_instructions.json` and are copied
into each experiment's `instructions.json`:

1. Nearest visible uncollected food takes priority.
2. Without visible food, target the nearest unvisited grid-cell centre.
3. Keep that exploration target until its centre is reached or food interrupts.
4. After all tiles are visited, target the least recently occupied tile.
5. Neural outputs select heading within 60 degrees of the target and throttle
   between 20% and 100% of maximum, enforcing movement toward the target.

Target selection, map tracking, goal persistence and movement constraints are
PROGRAMMED. Evolution only optimises the 170-weight controller's heading offset
and throttle. Its inputs include goal direction/distance, forage/explore mode,
own velocity and boundary rays. No unseen food or raw 64-cell map enters the NN.
The same rule selector with direct full-speed steering is a scripted reference;
it is not an optimal search algorithm. Charts separate the unevolved hybrid,
evolved steering and scripted reference. Food collection is the only reward.
Replays show the purple current goal and forage/explore status over visited tiles.
Do not describe this experiment as evolution inventing the foraging strategy.

Restore complex terrain for the hybrid controller:

```powershell
.\.venv\Scripts\python.exe -u evolution/single_pellet.py --hybrid-test --hybrid-arena terrain --pellet-counts 12 --generations 20 --seeds 5 --output results/hybrid-terrain-20
```

Uses the exported broad-outcrops map, the shared swept-circle collision solver,
and scattered food in its largest body-accessible region. Exploration targets
use a clear representative point in each tile; tiles without reachable sites are
ignored. Food remains visible through obstacles. Target priorities, persistent
goals, nearest-unvisited selection, minimum throttle and 60-degree steering cone
remain programmed rules. No pathfinding is added: the cone can prevent detours
that require moving away from a goal. The direct scripted reference uses the same
goals and collision rules and may also get stuck. This is an evaluation of the
current hybrid on terrain, not a claim of generally competent navigation. Data,
instructions, terrain, confidence-band graphs and purple-goal replays are saved.

Longer inspection of saved hybrid controllers, with no retraining:

```powershell
.\.venv\Scripts\python.exe -u evolution/replay_foraging.py results/hybrid-terrain-20 --seconds 120 --output results/hybrid-terrain-long-next
```

Uses the first three saved controller seeds and the first held-out start of each,
without selecting successes. Outputs include `long-replays.gif` (4x playback),
`viewer.html` (standalone player with 1x/2x/4x/8x, pause and seek), full replay JSON,
per-example metrics and a food/coverage-over-time figure. Coverage is the fraction
of body-accessible grid tiles entered by the centre, not all visible ground.
A red outline indicates that the centre stayed within one cell of its current
position during the previous two seconds while food remains. Contact counters
count collision-solver iterations; they are not unique wall encounters. These
are longer evaluations of ten-second-trained controllers, not longer training.

Hybrid stuck recovery is now explicitly specified in `hybrid_instructions.json`:
when the body stays within one cell for two seconds, cool down the current target
for 20 seconds. Temporarily override food priority and follow a collision-clear
local escape waypoint directly at 60% speed, for up to three seconds or until
reached. The waypoint is selected from 16 local directions out to 12 cells,
favouring clearance and backing away from the blocked goal. Normal NN steering
then resumes. Recovery is scripted, not evolved, and does not implement routing.

To apply the new rule to the SAME saved controllers and long starts:

```powershell
.\.venv\Scripts\python.exe -u evolution/replay_foraging.py results/hybrid-terrain-20 --seconds 120 --enable-recovery --output results/hybrid-terrain-recovery-next
```

Without `--enable-recovery`, saved old runs preserve their original recovery-off
setting. New hybrid training uses the new rule by default. The replay comparison
therefore isolates a programmed-rule intervention without retraining. Recordings
include recovery counts and `recover` mode, and save the applied instruction spec.

Larger paired recovery benchmark, using saved controllers without retraining:

```powershell
.\.venv\Scripts\python.exe -u evolution/benchmark_recovery.py results/hybrid-terrain-20 --starts 32 --seconds 120 --output results/recovery-benchmark-next
```

Uses all five controller seeds on 32 fresh arrangements each: 160 encounters per
condition, paired recovery off/on. Food and visited-tile trajectories are sampled
once per second. `recovery-means-ci.png` shows means and pointwise 95% bootstrap
confidence bands; `paired-differences.png` shows on-minus-off effects. Bootstrap
resamples whole controller seeds after averaging their starts, preserving paired
conditions. Thus 160 encounters are NOT treated as 160 independent evolved
controllers: only five independent controller seeds support those intervals.
Coverage percentages use reachable grid tiles; raw visited counts include the
starting tile. The benchmark is conditional on this single map. Encounter CSV,
seed-average trajectories and compressed per-encounter curves are saved so plots
can be changed without rerunning simulation.
