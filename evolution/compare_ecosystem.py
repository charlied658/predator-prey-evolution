"""Numeric-first, budget-matched comparisons of small evolutionary algorithms."""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import random
import statistics
import subprocess
import time
from pathlib import Path

from config import GODOT_PATH, PROJECT_PATH
from ecosystem_numeric import evaluate
from evolve_ecosystem import BOUNDS, DEFAULTS, fitness, mutate, settings_for, write_json

ALGORITHMS = ('es', 'ga', 'de', 'random')
NAMES = {'es': 'Mutation-only ES', 'ga': 'Genetic algorithm', 'de': 'Differential evolution', 'random': 'Random search (control)'}


def initial_population(baseline, size, seed, mutation):
    rng = random.Random(seed)
    return [baseline.copy()] + [mutate(baseline, rng, mutation) for _ in range(size - 1)]


def offspring(algorithm, population, scores, rng, mutation):
    """All methods propose exactly population_size - 1 new candidates per generation."""
    elite = max(range(len(scores)), key=scores.__getitem__)
    if algorithm == 'es':
        return [mutate(population[elite], rng, mutation) for _ in range(len(population) - 1)], []
    if algorithm == 'ga':
        def tournament():
            contestants = rng.sample(range(len(population)), min(2, len(population)))
            return population[max(contestants, key=scores.__getitem__)]
        children = []
        for _ in range(len(population) - 1):
            a, b = tournament(), tournament()
            crossed = {key: a[key] if rng.random() < 0.5 else b[key] for key in a}
            children.append(mutate(crossed, rng, mutation))
        return children, []
    if algorithm == 'random':
        return [{key: round(rng.uniform(*BOUNDS[key]), 6) for key in population[0]}
                for _ in range(len(population) - 1)], []
    if algorithm != 'de' or len(population) < 4:
        raise ValueError('Differential evolution requires at least four candidates')
    targets = [index for index in range(len(population)) if index != elite]
    children = []
    for target in targets:
        a, b, c = rng.sample([i for i in range(len(population)) if i != target], 3)
        forced = rng.choice(list(population[target]))
        trial = {}
        for key in population[target]:
            value = population[target][key]
            if key == forced or rng.random() < 0.7:
                value = population[a][key] + 0.8 * (population[b][key] - population[c][key])
            low, high = BOUNDS[key]
            trial[key] = round(max(low, min(high, value)), 6)
        children.append(trial)
    return children, targets


def select_population(algorithm, population, scores, children, child_scores, targets):
    if algorithm == 'de':
        next_population = copy.deepcopy(population)
        next_scores = scores[:]
        for child, score, target in zip(children, child_scores, targets):
            if score >= scores[target]:
                next_population[target], next_scores[target] = child.copy(), score
        return next_population, next_scores
    elite = max(range(len(scores)), key=scores.__getitem__)
    return [population[elite].copy()] + children, [scores[elite]] + child_scores


class Evaluator:
    def __init__(self, base, role, seeds, workers, folder, stage):
        self.base, self.role, self.seeds = base, role, seeds
        self.workers, self.folder, self.stage = workers, folder, stage
        self.cache = {}
        self.episodes = 0

    def rows(self, genomes):
        missing = {}
        for genome in genomes:
            for seed in self.seeds:
                key = (tuple(sorted(genome.items())), seed)
                if key not in self.cache and key not in missing:
                    settings = settings_for(self.base, self.role, genome)
                    settings['seed'] = seed
                    missing[key] = settings
        results = evaluate(list(missing.values()), self.workers)
        with (self.folder / 'episodes.jsonl').open('a', encoding='utf-8') as file:
            for (key, settings), result in zip(missing.items(), results):
                self.cache[key] = result
                self.episodes += 1
                file.write(json.dumps({'stage': self.stage, 'seed': settings['seed'],
                                       'genes': dict(key[0]), 'result': result}, allow_nan=False) + '\n')
        return [[self.cache[(tuple(sorted(genome.items())), seed)] for seed in self.seeds]
                for genome in genomes]

    def scores(self, genomes):
        return [statistics.mean(fitness(row, self.role) for row in group) for group in self.rows(genomes)]


def write_history(folder, rows):
    if rows:
        with (folder / 'history.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run_method(args, base, baseline, algorithm, repeat, all_rows):
    seed = args.seed + repeat
    folder = args.output / algorithm / f'run-{repeat:03d}'
    folder.mkdir(parents=True)
    train = Evaluator(base, args.role, args.train_seeds, args.workers, folder, 'training')
    validation = Evaluator(base, args.role, args.validation_seeds, args.workers, folder, 'validation')
    population = initial_population(baseline, args.population, seed, args.mutation)
    rng = random.Random(seed + 1000000)  # Independent of initial-population construction.
    started = time.perf_counter()
    scores = train.scores(population)
    original_training = train.scores([baseline])[0]
    original_validation = validation.scores([baseline])[0]
    requested = args.population
    local_rows = []
    for generation in range(args.generations + 1):
        if generation:
            children, targets = offspring(algorithm, population, scores, rng, args.mutation)
            child_scores = train.scores(children)
            population, scores = select_population(algorithm, population, scores, children, child_scores, targets)
            requested += len(children)
        champion = max(range(len(scores)), key=scores.__getitem__)
        genes = population[champion]
        # Validation is recorded only; it never selects parents or replaces the training winner.
        validation_score = validation.scores([genes])[0]
        training_rows = train.rows([genes])[0]
        validation_rows = validation.rows([genes])[0]
        row = {'algorithm': algorithm, 'run': repeat, 'optimizer_seed': seed, 'generation': generation,
               'candidate_evaluations': requested, 'training_fitness': scores[champion],
               'population_mean_fitness': statistics.mean(scores), 'validation_fitness': validation_score,
               'baseline_training_fitness': original_training, 'baseline_validation_fitness': original_validation,
               'training_episodes': train.episodes, 'validation_episodes': validation.episodes,
               'wall_seconds': time.perf_counter() - started,
               'training_captures': statistics.mean(r['captures'] for r in training_rows),
               'training_food_eaten': statistics.mean(r['food_eaten'] for r in training_rows),
               'training_starvations': statistics.mean(r['starvations'][args.role] for r in training_rows),
               'validation_captures': statistics.mean(r['captures'] for r in validation_rows),
               'validation_food_eaten': statistics.mean(r['food_eaten'] for r in validation_rows),
               'validation_starvations': statistics.mean(r['starvations'][args.role] for r in validation_rows), **genes}
        local_rows.append(row)
        all_rows.append(row)
        write_history(args.output, all_rows)
        write_history(folder, local_rows)
        config = settings_for(base, args.role, genes)
        config['seed'] = args.train_seeds[0]
        config['evolution_label'] = f'{NAMES[algorithm]} | generation {generation} | training {scores[champion]:g}'
        write_json(folder / f'generation-{generation:04d}.json', config)
        write_json(folder / 'best.json', config)
        validation_config = copy.deepcopy(config)
        validation_config['seed'] = args.validation_seeds[0]
        write_json(folder / 'validation.json', validation_config)
        write_json(folder / f'population-{generation:04d}.json', [
            {'genes': genome, 'fitness': score} for genome, score in zip(population, scores)])
        print(f'{algorithm:6s} run {repeat + 1}/{args.runs} gen {generation}/{args.generations} | '
              f'train {scores[champion]:6.2f} | validation {validation_score:6.2f} | slots {requested}', flush=True)
    return local_rows[-1]


def compare(args):
    if args.population < 2 or args.generations < 1 or args.runs < 1 or args.workers < 1:
        raise ValueError('Need population >= 2 and generations/runs/workers >= 1')
    if 'de' in args.algorithms and args.population < 4:
        raise ValueError('Differential evolution needs --population >= 4')
    if len(set(args.algorithms)) != len(args.algorithms):
        raise ValueError('Algorithms must be distinct')
    if any(algorithm not in ALGORITHMS for algorithm in args.algorithms):
        raise ValueError('Unknown algorithm')
    if set(args.train_seeds) & set(args.validation_seeds):
        raise ValueError('Training and validation seeds must not overlap')
    if any(len(seeds) != len(set(seeds)) for seeds in [args.train_seeds, args.validation_seeds]):
        raise ValueError('Scenario seeds must be distinct within each suite')
    if not math.isfinite(args.duration) or args.duration <= 0 or not math.isfinite(args.mutation) or not 0 < args.mutation <= 1:
        raise ValueError('duration must be positive; mutation must be in (0, 1]')
    if not args.no_plots and importlib.util.find_spec('matplotlib') is None:
        raise ValueError('Plotting requires Matplotlib: install requirements.txt, use .venv/bin/python, or pass --no-plots for CSV/JSON only')
    slots = args.population + args.generations * (args.population - 1)
    per_run = slots * len(args.train_seeds) + (args.generations + 2) * len(args.validation_seeds)
    maximum = len(args.algorithms) * args.runs * per_run
    if maximum > args.max_episodes:
        raise ValueError(f'Plan needs at most {maximum} episodes; reduce settings or explicitly raise --max-episodes {maximum}')
    base = json.loads(args.config.read_text())
    base['duration'] = args.duration
    if base['predators'] <= 0 or base['prey'] <= 0:
        raise ValueError('Comparison requires both species')
    baseline = {**DEFAULTS[args.role], **base.get('behavior', {}).get(args.role, {})}
    if set(baseline) != set(DEFAULTS[args.role]):
        raise ValueError('Unsupported baseline genes')
    for key, value in baseline.items():
        low, high = BOUNDS[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'Baseline gene {key} must be in [{low}, {high}]')
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output is not empty; choose a new output folder')
    args.output.mkdir(parents=True, exist_ok=True)
    sources = ['scripts/ecosystem.gd', 'scripts/agent.gd', 'scripts/ecosystem_numeric.gd',
               'evolution/compare_ecosystem.py', 'evolution/evolve_ecosystem.py', 'evolution/ecosystem_numeric.py']
    metadata = {'settings': base, 'role': args.role, 'algorithms': args.algorithms, 'population': args.population,
                'generations': args.generations, 'runs': args.runs, 'mutation': args.mutation,
                'optimizer_seeds': [args.seed + index for index in range(args.runs)],
                'train_seeds': args.train_seeds, 'validation_seeds': args.validation_seeds,
                'maximum_episodes': maximum, 'candidate_slots_per_method_run': slots,
                'workers': args.workers, 'bounds': BOUNDS,
                'fitness': 'captures - 2 * predator_starvations' if args.role == 'predator'
                else 'food_eaten - 2 * captures - 2 * prey_starvations',
                'godot_version': subprocess.check_output([GODOT_PATH, '--version'], text=True).strip(),
                'source_sha256': {name: hashlib.sha256((Path(PROJECT_PATH) / name).read_bytes()).hexdigest() for name in sources}}
    write_json(args.output / 'experiment.json', metadata)
    baseline_config = settings_for(base, args.role, baseline)
    baseline_config['seed'] = args.train_seeds[0]
    write_json(args.output / 'baseline.json', baseline_config)
    print(f'Numeric comparison: {len(args.algorithms)} methods x {args.runs} optimizer runs; '
          f'at most {maximum} episodes x {args.duration:g}s. No rendering.', flush=True)
    all_rows, finals = [], []
    started = time.perf_counter()
    for algorithm in args.algorithms:
        for repeat in range(args.runs):
            finals.append(run_method(args, base, baseline, algorithm, repeat, all_rows))
    summary = {'wall_seconds': time.perf_counter() - started, 'maximum_episodes': maximum,
               'actual_episodes': sum(row['training_episodes'] + row['validation_episodes'] for row in finals),
               'algorithms': {}}
    for algorithm in args.algorithms:
        group = [row for row in finals if row['algorithm'] == algorithm]
        summary['algorithms'][algorithm] = {
            'training_mean': statistics.mean(row['training_fitness'] for row in group),
            'validation_mean': statistics.mean(row['validation_fitness'] for row in group),
            'validation_std': statistics.stdev(row['validation_fitness'] for row in group) if len(group) > 1 else None,
            'mean_wall_seconds': statistics.mean(row['wall_seconds'] for row in group)}
    write_json(args.output / 'summary.json', summary)
    if not args.no_plots:
        from plot_ecosystem import make_plots
        make_plots(args.output)
    print(f'Saved numeric data to {args.output / "history.csv"}', flush=True)
    print(f'Finished {summary["actual_episodes"]} actual episodes in {summary["wall_seconds"]:.2f}s (before plotting).', flush=True)
    return all_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('ecosystem.json'))
    parser.add_argument('--role', choices=['prey', 'predator'], default='prey')
    parser.add_argument('--algorithms', nargs='+', choices=ALGORITHMS, default=['es', 'ga', 'de'])
    parser.add_argument('--population', type=int, default=4)
    parser.add_argument('--generations', type=int, default=3)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--duration', type=float, default=10)
    parser.add_argument('--seed', type=int, default=100, help='First optimizer seed, separate from scenario seeds')
    parser.add_argument('--train-seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--validation-seeds', nargs='+', type=int, default=[10042])
    parser.add_argument('--mutation', type=float, default=0.15)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--max-episodes', type=int, default=200)
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('results/ecosystem-comparison'))
    args = parser.parse_args()
    try:
        compare(args)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
