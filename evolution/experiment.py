"""Reproducible fixed-versus-diverse prey study. No third-party packages."""
import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path

from godot_interface import evaluate_batch

GENOME_LENGTH = 66


def genome(rng):
    return [rng.uniform(-1, 1) for _ in range(GENOME_LENGTH)]


def scenario(rng):
    while True:
        p = [rng.uniform(-12, 12), rng.uniform(-12, 12)]
        q = [rng.uniform(-12, 12), rng.uniform(-12, 12)]
        if 5 <= math.dist(p, q) <= 20:
            return {"predator_start": p, "prey_start": q}


def suite(seed, count):
    rng = random.Random(seed)
    cases = []
    for kind in ["neural", "stationary", "straight", "flee"]:
        for _ in range(count):
            cases.append({**scenario(rng), "prey_controller": kind,
                          "prey_genome": genome(rng) if kind == "neural" else []})
    return cases


def evaluate(population, cases, workers):
    encounters = [{**case, "predator_genome": individual}
                  for individual in population for case in cases]
    rows = evaluate_batch(encounters, workers)
    size = len(cases)
    return [rows[i:i + size] for i in range(0, len(rows), size)]


def fitness(row, shaping):
    # Capture remains dominant within each encounter; shaping only distinguishes failures.
    if row["captured"] or not shaping:
        return row["predator_fitness"]
    return 1 / (1 + row["minimum_distance"])


def metrics(rows):
    captures = [row["time"] for row in rows if row["captured"]]
    return {
        "capture_rate": statistics.mean(row["captured"] for row in rows),
        "capture_time": statistics.mean(captures) if captures else None,
        "timeout_rate": 1 - len(captures) / len(rows),
        "minimum_distance": statistics.mean(row["minimum_distance"] for row in rows),
        "predator_boundary_fraction": statistics.mean(row["predator_boundary_time"] / row["time"] for row in rows),
        "prey_boundary_fraction": statistics.mean(row["prey_boundary_time"] / row["time"] for row in rows),
        "predator_path_length": statistics.mean(row["predator_path_length"] for row in rows),
    }


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def breed(population, scores, rng):
    ranked = sorted(range(len(population)), key=lambda i: scores[i], reverse=True)
    children = [population[i][:] for i in ranked[:2]]
    def select():
        candidates = rng.sample(range(len(population)), min(3, len(population)))
        return population[max(candidates, key=lambda i: scores[i])]
    while len(children) < len(population):
        a, b = select(), select()
        child = [x if rng.random() < 0.5 else y for x, y in zip(a, b)]
        children.append([x + rng.gauss(0, 0.2) if rng.random() < 0.1 else x for x in child])
    return children


def train(args, condition, seed, validation, test):
    folder = args.output / f"{condition}-seed-{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / "checkpoint.json"
    rng = random.Random(seed)
    population = [genome(rng) for _ in range(args.population)]
    opponent_rng = random.Random(seed + 1_000_000)
    opponents = [genome(opponent_rng) for _ in range(args.pool_size)]
    history = []
    champion = None
    champion_score = -1
    start_generation = 0
    if args.resume and checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        population = state["population"]
        def tuples(value):
            return tuple(tuples(x) for x in value) if isinstance(value, list) else value
        rng.setstate(tuples(state["rng_state"]))
        history = state["history"]
        champion = state["champion"]
        champion_score = state["champion_score"]
        start_generation = state["next_generation"]
    for generation in range(start_generation, args.generations):
        cases_rng = random.Random(seed * 100_003 + generation + 2_000_000)
        cases = []
        for episode in range(args.episodes):
            # Same starts in both conditions, opponent pool sampled on a separate RNG.
            opponent_index = random.Random(seed * 100_003 + generation * args.episodes + episode).randrange(args.pool_size)
            cases.append({**scenario(cases_rng), "prey_genome": opponents[0 if condition == "fixed" else opponent_index]})
        evaluated = evaluate(population, cases, args.workers)
        scores = [statistics.mean(fitness(row, args.shaping) for row in rows) for rows in evaluated]
        winner = max(range(len(population)), key=lambda i: scores[i])
        # Selection of the saved champion uses validation only, never the final test suite.
        validation_rows = evaluate([population[winner]], validation, args.workers)[0]
        validation_score = metrics(validation_rows)["capture_rate"]
        if champion is None or validation_score > champion_score:
            champion = population[winner][:]
            champion_score = validation_score
        history.append({"generation": generation + 1,
                        "training_encounters": (generation + 1) * args.population * args.episodes,
                        "total_selection_encounters": (generation + 1) * (args.population * args.episodes + len(validation)),
                        "best_training_fitness": max(scores), "mean_training_fitness": statistics.mean(scores),
                        "validation_capture_rate": validation_score,
                        "champion_validation_capture_rate": champion_score})
        population = breed(population, scores, rng)
        write_json(checkpoint, {"next_generation": generation + 1, "population": population,
                               "rng_state": rng.getstate(), "history": history,
                               "champion": champion, "champion_score": champion_score})
        print(f"{condition} seed={seed} generation={generation + 1} validation={validation_score:.3f}", flush=True)
    final_rows = evaluate([champion], test, args.workers)[0]
    raw = [{**case, **row} for case, row in zip(test, final_rows)]
    write_json(folder / "test_encounters.json", raw)
    write_json(folder / "best_predator.json", {
        "genome": champion, "prey_genome": opponents[0],
        "replay_encounter": {**test[0], "predator_genome": champion},
        "fitness_history": [row["best_training_fitness"] for row in history],
        "condition": condition, "seed": seed})
    report = {"condition": condition, "seed": seed, **metrics(final_rows),
              "by_opponent": {kind: metrics([row for case, row in zip(test, final_rows) if case["prey_controller"] == kind])
                              for kind in ["neural", "stationary", "straight", "flee"]}, "history": history}
    write_json(folder / "report.json", report)
    return report


def bootstrap(values):
    if len(values) < 2:
        return None
    rng = random.Random(317)
    samples = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(5000))
    return [samples[125], samples[4874]]


def summarize(output, reports):
    summary = {}
    for condition in ["fixed", "diverse"]:
        values = [r["capture_rate"] for r in reports if r["condition"] == condition]
        summary[condition] = {"mean_test_capture_rate": statistics.mean(values),
                              "run_count": len(values), "bootstrap_95_ci": bootstrap(values)}
    paired = [next(r["capture_rate"] for r in reports if r["condition"] == "diverse" and r["seed"] == seed)
              - next(r["capture_rate"] for r in reports if r["condition"] == "fixed" and r["seed"] == seed)
              for seed in sorted({r["seed"] for r in reports})]
    summary["paired_diverse_minus_fixed"] = {"mean": statistics.mean(paired), "bootstrap_95_ci": bootstrap(paired)}
    write_json(output / "summary.json", summary)
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as file:
        keys = ["condition", "seed", "capture_rate", "capture_time", "timeout_rate", "minimum_distance", "predator_boundary_fraction", "prey_boundary_fraction", "predator_path_length"]
        writer = csv.DictWriter(file, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(reports)
    with (output / "learning_curves.csv").open("w", newline="", encoding="utf-8") as file:
        keys = ["condition", "seed", *reports[0]["history"][0]]
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for report in reports:
            for row in report["history"]:
                writer.writerow({"condition": report["condition"], "seed": report["seed"], **row})
    # Standalone vector plot: each point is a run's final held-out result.
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="720" height="400" viewBox="0 0 720 400">',
           '<rect width="720" height="400" fill="white"/>',
           '<text x="45" y="28" font-family="sans-serif" font-size="18">Held-out capture rate by independent training run</text>']
    for tick in range(6):
        y = 330 - tick * 54
        svg.append(f'<path d="M70 {y} H670" stroke="#ddd"/><text x="30" y="{y + 5}" font-family="sans-serif">{tick / 5:.1f}</text>')
    for condition, x, color in [("fixed", 240, "#2563eb"), ("diverse", 500, "#db2777")]:
        subset = [r for r in reports if r["condition"] == condition]
        svg.append(f'<text x="{x - 30}" y="365" font-family="sans-serif">{condition}</text>')
        for i, report in enumerate(subset):
            dx = (i - (len(subset) - 1) / 2) * min(14, 150 / len(subset))
            svg.append(f'<circle cx="{x + dx}" cy="{330 - 270 * report["capture_rate"]}" r="5" fill="{color}"/>')
        average = summary[condition]["mean_test_capture_rate"]
        y = 330 - 270 * average
        svg.append(f'<path d="M{x - 40} {y} h80" stroke="{color}" stroke-width="3"/>')
        interval = summary[condition]["bootstrap_95_ci"]
        if interval:
            svg.append(f'<path d="M{x} {330 - 270 * interval[0]} V{330 - 270 * interval[1]}" stroke="{color}" stroke-width="2"/>')
    svg.append('<text x="70" y="390" font-family="sans-serif" font-size="12">Dots: runs. Bars: means. Vertical lines: bootstrap 95% intervals across runs.</text></svg>')
    (output / "capture_rates.svg").write_text("\n".join(svg), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/study"))
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--pool-size", type=int, default=8)
    parser.add_argument("--validation-per-kind", type=int, default=8)
    parser.add_argument("--test-per-kind", type=int, default=50)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shaping", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in ["runs", "population", "generations", "episodes", "pool_size", "validation_per_kind", "test_per_kind", "workers"]:
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if args.population < 3:
        parser.error("population must be at least 3")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != "resume"}
    config_path = args.output / "experiment.json"
    if config_path.exists():
        if not args.resume:
            parser.error("Output already contains a study; use a new directory or --resume")
        if json.loads(config_path.read_text()) != config:
            parser.error("Resume requires identical experiment settings")
    elif args.resume:
        parser.error("No experiment exists to resume")
    else:
        write_json(config_path, config)
    sources = [Path(__file__), Path(__file__).with_name("godot_interface.py"),
               Path(__file__).resolve().parents[1] / "scripts" / "agent.gd",
               Path(__file__).resolve().parents[1] / "scripts" / "simulation.gd"]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    hashes_path = args.output / "source_hashes.json"
    if hashes_path.exists() and json.loads(hashes_path.read_text()) != hashes:
        parser.error("Source changed since this study began; use a new output directory")
    write_json(hashes_path, hashes)
    validation = suite(9_000_001, args.validation_per_kind)
    test = suite(9_000_002, args.test_per_kind)
    write_json(args.output / "validation_suite.json", validation)
    write_json(args.output / "test_suite.json", test)
    started = time.perf_counter()
    reports = [train(args, condition, args.seed + run, validation, test)
               for run in range(args.runs) for condition in ["fixed", "diverse"]]
    summarize(args.output, reports)
    write_json(args.output / "timing.json", {"invocation_seconds": time.perf_counter() - started})


if __name__ == "__main__":
    main()
