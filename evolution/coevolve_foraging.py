"""Test whether identical lineages specialize in a symmetric Godot arena."""
import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path

from ecosystem_numeric import evaluate
from evolve_ecosystem import write_json


GENES = 70
POPULATION = 4


def mutate(parent, rng):
    weights = parent['weights'].copy()
    body = parent['body'].copy()
    for index in rng.sample(range(GENES), 4):
        weights[index] += rng.gauss(0, 0.35)
    index = rng.randrange(len(body))
    body[index] = min(1, max(0, body[index] + rng.gauss(0, 0.15)))
    return {'weights': weights, 'body': body}


def settings(base, a, b, seed, swapped):
    return {**base, 'mode': 'food_competition', 'genomes': [a['weights'], b['weights']],
            'body_genes': [a['body'], b['body']], 'swimming': True,
            'seed': seed, 'swap_starts': swapped}


def matches(base, pairs, seed, workers):
    jobs = [settings(base, a, b, seed, swapped)
            for a, b in pairs for swapped in (False, True)]
    return evaluate(jobs, workers)


def plot(output, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    panels = [
        ('head_to_head', 'collected', 'Pellets against each other'),
        ('solo', 'collected', 'Pellets when alone'),
        ('solo', 'mean_speed', 'Movement speed when alone (units/s)'),
        ('solo', 'turn_radians_per_second', 'Path turning when alone (rad/s)'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, layout='constrained')
    for ax, (matchup, metric, title) in zip(axes.flat, panels):
        for side, color in (('A', '#be493f'), ('B', '#218780')):
            by_generation = {}
            for row in rows:
                if row['lineage'] == side and row['matchup'] == matchup:
                    by_generation.setdefault(row['generation'], []).append(row[metric])
            ax.plot(list(by_generation), [statistics.mean(values) for values in by_generation.values()],
                    marker='o', color=color, label=f'Lineage {side}')
        ax.set(title=title, xlabel='Generation')
        ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.savefig(output / 'specialization.png', dpi=160)
    plt.close(fig)


def experiment(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('Output directory must be empty')
    base = json.loads(Path(__file__).with_name('single_pair.json').read_text())
    base.update(duration=args.duration, predator_home=[-7, 0], prey_home=[7, 0],
                rocks=14, food_pellets=28)
    initializer = random.Random(args.seed)
    initial = [{'weights': [initializer.gauss(0, 0.7) for _ in range(GENES)],
                'body': [0.5] * 4} for _ in range(POPULATION)]
    streams = [random.Random(initializer.getrandbits(64)) for _ in range(2)]
    benchmark_seed = args.seed + 10000
    probe_seeds = [benchmark_seed, benchmark_seed + 1]
    write_json(args.output / 'experiment.json', {
        'optimizer_seed': args.seed, 'training_seed': args.seed, 'benchmark_seed': benchmark_seed,
        'generations': args.generations, 'population': POPULATION, 'settings': base,
        'starting_population': initial, 'solo_probe_seeds': probe_seeds,
    })
    champions = [initial[0], initial[0]]
    pools = [initial, initial]
    rows = []
    genome_history = []
    for generation in range(args.generations + 1):
        if generation:
            scores = [[0] * POPULATION for _ in range(2)]
            results = iter(matches(base, [(a, b) for a in pools[0] for b in pools[1]],
                                   args.seed, args.workers))
            for a in range(POPULATION):
                for b in range(POPULATION):
                    for _ in range(2):
                        payoff_a, payoff_b = next(results)['payoffs']
                        scores[0][a] += payoff_a
                        scores[1][b] += payoff_b
            champions = [pool[max(range(POPULATION), key=score.__getitem__)]
                         for pool, score in zip(pools, scores)]
            pools = [[parent] + [mutate(parent, rng) for _ in range(POPULATION - 1)]
                     for parent, rng in zip(champions, streams)]
        a, b = champions
        genome_history.append({
            'generation': generation,
            'weight_distance': math.dist(a['weights'], b['weights']),
            'distinct_weights': sum(x != y for x, y in zip(a['weights'], b['weights'])),
            'body_A': a['body'], 'body_B': b['body'],
            'body_distance': math.dist(a['body'], b['body']),
        })
        outcomes = matches(base, [tuple(champions)], benchmark_seed, args.workers)
        for swapped, match in enumerate(outcomes):
            for lineage, outcome, payoff in zip(('A', 'B'), match['lineages'], match['payoffs']):
                rows.append({'generation': generation, 'matchup': 'head_to_head', 'lineage': lineage,
                             'seed': benchmark_seed, 'swapped': swapped, 'collected': outcome['collected'],
                             'payoff': payoff, 'mean_speed': None, 'turn_radians_per_second': None})
        probes = [
            (lineage, {**settings(base, genome, genome, seed, swapped), 'solo_probe': True})
            for lineage, genome in zip(('A', 'B'), champions)
            for seed in probe_seeds for swapped in (False, True)
        ]
        for (lineage, config), match in zip(probes, evaluate([job for _, job in probes], args.workers)):
            outcome = match['lineages'][0]
            rows.append({'generation': generation, 'matchup': 'solo', 'lineage': lineage,
                         'seed': config['seed'], 'swapped': int(config['swap_starts']),
                         'collected': outcome['collected'], 'payoff': None,
                         'mean_speed': outcome['mean_speed'],
                         'turn_radians_per_second': outcome['turn_radians_per_second']})
        for swapped in (False, True):
            suffix = '-swapped' if swapped else ''
            write_json(args.output / f'generation-{generation:03d}{suffix}.json',
                       settings(base, *champions, benchmark_seed, swapped))
        print(f'Generation {generation}/{args.generations} | '
              f'held-out pellets {[[r["collected"] for r in job["lineages"]] for job in outcomes[:2]]}',
              flush=True)
    with (args.output / 'history.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output / 'genome_history.json', genome_history)
    body_swap(args, base, champions, probe_seeds)
    if not args.no_plots:
        plot(args.output, rows)


def body_swap(args, base, champions, probe_seeds):
    jobs, labels = [], []
    for controller, weights_owner in zip(('A', 'B'), champions):
        for body, body_owner in zip(('A', 'B'), champions):
            organism = {'weights': weights_owner['weights'], 'body': body_owner['body']}
            for seed in probe_seeds:
                for swapped in (False, True):
                    job = {**settings(base, organism, organism, seed, swapped), 'solo_probe': True}
                    jobs.append(job)
                    labels.append({'controller': controller, 'body': body, 'seed': seed,
                                   'swapped': swapped})
                    if seed == probe_seeds[0] and not swapped:
                        write_json(args.output / f'controller-{controller}-body-{body}.json', job)
    rows = [{**label, **result['lineages'][0]}
            for label, result in zip(labels, evaluate(jobs, args.workers))]
    with (args.output / 'body_swap.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--generations', type=int, default=6)
    parser.add_argument('--duration', type=float, default=20)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--output', type=Path, default=Path('results/coevolve-swimming'))
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    if min(args.generations, args.workers) < 1 or not 0 < args.duration < float('inf'):
        parser.error('Use positive generations, workers, and finite duration')
    experiment(args)


if __name__ == '__main__':
    main()
