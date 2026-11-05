"""Minimal evolution: keep the best scripted controller and mutate copies."""
import argparse
import copy
import csv
import json
import math
import random
import subprocess
import time
from pathlib import Path

from config import GODOT_PATH, PROJECT_PATH

DEFAULTS = {
    'predator': {'target_weight': 1.0, 'avoid_weight': 1.8, 'wander_weight': 0.5},
    'prey': {'target_weight': 2.5, 'avoid_weight': 1.8, 'wander_weight': 0.5, 'food_threshold': 85.0},
}
BOUNDS = {'target_weight': (0.25, 5.0), 'avoid_weight': (0.25, 4.0),
          'wander_weight': (0.05, 1.5), 'food_threshold': (40.0, 98.0)}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def mutate(parent, rng, amount=0.15):
    """A small Gaussian change to each gene, bounded to useful values."""
    child = {}
    for name, value in parent.items():
        low, high = BOUNDS[name]
        child[name] = round(max(low, min(high, value + rng.gauss(0, amount * (high - low)))), 6)
    return child


def fitness(result, role):
    if role == 'predator':
        return result['captures'] - 2 * result['starvations']['predator']
    return result['food_eaten'] - 2 * result['captures'] - 2 * result['starvations']['prey']


def settings_for(base, role, genome):
    settings = copy.deepcopy(base)
    settings.setdefault('behavior', {})[role] = genome.copy()
    settings.pop('evolution_label', None)
    return settings


def run_episode(settings, folder, index):
    source = folder / f'{index:04d}.config.json'
    target = folder / f'{index:04d}.summary.json'
    write_json(source, settings)
    command = [GODOT_PATH, '--headless', '--fixed-fps', '60', '--path', PROJECT_PATH,
               '--log-file', str(folder / f'{index:04d}.log'), 'res://ecosystem.tscn', '--',
               '--ecosystem-config', str(source), '--ecosystem-output', str(target)]
    process = subprocess.run(command, capture_output=True, text=True,
                             timeout=max(60, settings['duration'] * 5))
    if process.returncode or not target.exists():
        raise RuntimeError(f'Godot episode failed ({process.returncode}):\n{process.stdout}\n{process.stderr}')
    return json.loads(target.read_text())


def train(args):
    if args.population < 2 or args.generations < 1:
        raise ValueError('Need population >= 2 and generations >= 1')
    if not math.isfinite(args.duration) or args.duration <= 0:
        raise ValueError('duration must be positive and finite')
    if not math.isfinite(args.mutation) or not 0 < args.mutation <= 1:
        raise ValueError('mutation must be in (0, 1]')
    base = json.loads(args.config.read_text())
    base.update(seed=args.seed, duration=args.duration)
    if base['predators'] <= 0 or base['prey'] <= 0:
        raise ValueError('Evolution requires at least one predator and one prey')
    baseline = {**DEFAULTS[args.role], **base.get('behavior', {}).get(args.role, {})}
    if set(baseline) != set(DEFAULTS[args.role]):
        raise ValueError('Starting behavior contains unsupported genes for the selected role')
    for key, value in baseline.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= (100 if key == 'food_threshold' else 6):
            raise ValueError(f'Invalid starting gene: {key}')
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output folder is not empty; choose a new --output folder')
    output.mkdir(parents=True, exist_ok=True)
    episodes = output / 'episodes'
    episodes.mkdir()
    write_json(output / 'experiment.json', {
        'role': args.role, 'population': args.population, 'generations': args.generations,
        'mutation': args.mutation, 'settings': base, 'bounds': BOUNDS,
        'method': 'one unchanged winner plus Gaussian-mutated copies; one fixed world seed',
        'fitness': 'captures - 2 * predator_starvations' if args.role == 'predator'
                   else 'food_eaten - 2 * captures - 2 * prey_starvations',
    })
    cache = {}
    rng = random.Random(args.seed)
    history = []
    start = time.perf_counter()

    def evaluate(genome):
        # The unchanged elite needs no rerun because seed, opponents, and settings stay fixed.
        key = tuple(sorted(genome.items()))
        if key not in cache:
            result = run_episode(settings_for(base, args.role, genome), episodes, len(cache))
            cache[key] = (fitness(result, args.role), result)
        return cache[key]

    def save_winner(generation, genome, score, result):
        settings = settings_for(base, args.role, genome)
        settings['evolution_label'] = f'{args.role} evolution | generation {generation} | fitness {score:g}'
        write_json(output / f'generation-{generation:04d}.json', settings)
        write_json(output / 'best.json', settings)
        row = {'generation': generation, 'fitness': score, 'captures': result['captures'],
               'food_eaten': result['food_eaten'], 'predator_starvations': result['starvations']['predator'],
               'prey_starvations': result['starvations']['prey'], **genome}
        history.append(row)
        with (output / 'history.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)

    budget = 1 + args.generations * (args.population - 1)
    print(f'Evolving {args.role}: {args.population} candidates, {args.generations} generations; '
          f'at most {budget} episodes of {args.duration:g}s on seed {args.seed}.', flush=True)
    winner = baseline.copy()
    score, result = evaluate(winner)
    save_winner(0, winner, score, result)
    write_json(output / 'baseline.json', settings_for(base, args.role, winner))
    print(f'Original behavior fitness: {score:g}', flush=True)
    for generation in range(1, args.generations + 1):
        population = [winner.copy()] + [mutate(winner, rng, args.mutation) for _ in range(args.population - 1)]
        evaluated = [evaluate(genome) for genome in population]
        best = max(range(len(population)), key=lambda index: evaluated[index][0])
        winner = population[best].copy()
        score, result = evaluated[best]
        # Keep all candidate genes/scores so selection is inspectable.
        write_json(output / f'population-{generation:04d}.json', [
            {'genes': genome, 'fitness': evaluation[0]} for genome, evaluation in zip(population, evaluated)])
        save_winner(generation, winner, score, result)
        print(f'Generation {generation}/{args.generations}: fitness {score:g}; genes {winner}', flush=True)
    write_json(output / 'timing.json', {'episodes_run': len(cache), 'wall_seconds': time.perf_counter() - start})
    print(f'Watch: python3 evolution/ecosystem.py --config {args.output / "best.json"}', flush=True)
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('ecosystem.json'))
    parser.add_argument('--role', choices=['predator', 'prey'], default='prey')
    parser.add_argument('--population', type=int, default=4)
    parser.add_argument('--generations', type=int, default=3)
    parser.add_argument('--duration', type=float, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mutation', type=float, default=0.15)
    parser.add_argument('--output', type=Path, default=Path('results/ecosystem-evolution-basic'))
    args = parser.parse_args()
    try:
        train(args)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
