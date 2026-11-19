"""Compare fixed, current, and archived opponents in a seeded zero-sum game."""
import argparse
import csv
import json
import random
import statistics
from pathlib import Path


METHODS = ('fixed', 'current', 'archive')
STRENGTHS = {'weak': 0.2, 'strong': 0.8}
PANEL_SIZE = 4
POPULATION_SIZE = 8


class Landscape:
    """Precompute each genome's indices into the small payoff tables."""

    def __init__(self, bits, seed):
        self.size = 1 << bits
        rng = random.Random(seed)
        self.own = []
        for _ in range(2):
            tables = [(rng.sample(range(bits), 3), self.entries(rng)) for _ in range(bits)]
            self.own.append([
                statistics.mean(values[self.index(genome, loci)] for loci, values in tables)
                for genome in range(self.size)
            ])
        self.interactions = []
        for i in range(2 * bits):
            count_a = 1 + i % 2
            a_loci = rng.sample(range(bits), count_a)
            b_loci = rng.sample(range(bits), 3 - count_a)
            values = self.entries(rng)
            self.interactions.append((
                [self.index(genome, a_loci) for genome in range(self.size)],
                [self.index(genome, b_loci) << count_a for genome in range(self.size)],
                values,
            ))

    @staticmethod
    def entries(rng):
        return [rng.uniform(-1, 1) for _ in range(8)]

    @staticmethod
    def index(genome, loci):
        return sum(((genome >> locus) & 1) << i for i, locus in enumerate(loci))

    def payoff(self, a, b, strength):
        independent = (self.own[0][a] - self.own[1][b]) / 2
        interaction = sum(table[x[a] | y[b]] for x, y, table in self.interactions)
        return (1 - strength) * independent + strength * interaction / len(self.interactions)


def checkpoint(game, strength, generation, a, b):
    """Evaluate unilateral best responses without feeding them into training."""
    worst_a = min(game.payoff(a, rival, strength) for rival in range(game.size))
    worst_b = -max(game.payoff(rival, b, strength) for rival in range(game.size))
    return {
        'generation': generation,
        'training_matches': generation * 2 * POPULATION_SIZE * PANEL_SIZE,
        'champion_a': a,
        'champion_b': b,
        'contemporary_payoff_a': game.payoff(a, b, strength),
        'worst_payoff_a': worst_a,
        'worst_payoff_b': worst_b,
        'pure_best_response_gap': -worst_a - worst_b,
    }


def run(game, strength, bits, seed, method, generations, interval, *, identical_start=False):
    initializer = random.Random(seed)
    pools = [initializer.sample(range(game.size), POPULATION_SIZE) for _ in range(2)]
    if identical_start:
        pools[1] = pools[0].copy()
    champions = [pool[0] for pool in pools]
    # Separate streams keep each method's mutation draws paired even when the
    # opponent-panel sampling differs.
    mutation_rng = [random.Random(initializer.getrandbits(64)) for _ in range(2)]
    panel_rng = [random.Random(initializer.getrandbits(64)) for _ in range(2)]
    fixed = [panel_rng[side].sample(pools[1 - side], PANEL_SIZE) for side in range(2)]
    archives = [[], []]
    rows = [checkpoint(game, strength, 0, *champions)]

    for generation in range(1, generations + 1):
        next_pools = []
        next_champions = []
        for side in range(2):
            opponent = 1 - side
            rng = panel_rng[side]
            panel = fixed[side]
            if method != 'fixed' and generation > 1:
                history = archives[opponent] if method == 'archive' else []
                historical_count = min(2, len(history))
                panel = rng.sample(pools[opponent], PANEL_SIZE - historical_count)
                panel += rng.sample(history, historical_count)
            incumbent = champions[side]
            candidates = [incumbent] + [
                incumbent ^ (1 << mutation_rng[side].randrange(bits))
                for _ in range(POPULATION_SIZE - 1)
            ]
            scores = []
            for candidate in candidates:
                if side == 0:
                    scores.append(sum(game.payoff(candidate, rival, strength) for rival in panel))
                else:
                    scores.append(-sum(game.payoff(rival, candidate, strength) for rival in panel))
            next_pools.append(candidates)
            next_champions.append(candidates[max(range(len(scores)), key=scores.__getitem__)])

        # Only strictly earlier incumbents enter the archive. Repeated genomes
        # can appear in both pools and archives, representing repeated slots.
        for side in range(2):
            archives[side].append(champions[side])
        pools, champions = next_pools, next_champions
        if generation % interval == 0 or generation == generations:
            rows.append(checkpoint(game, strength, generation, *champions))
    return rows


def save_plots(output, rows, matrices):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = {'fixed': '#737b86', 'current': '#2166ac', 'archive': '#c76b19'}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout='constrained')
    for axis, strength in zip(axes, STRENGTHS):
        for method in METHODS:
            selected = [row for row in rows if row['strength'] == strength and row['method'] == method]
            by_generation = {}
            for row in selected:
                by_generation.setdefault(row['generation'], []).append(row['pure_best_response_gap'])
            axis.plot(list(by_generation), [statistics.mean(values) for values in by_generation.values()],
                      marker='o', color=colors[method], label=method.capitalize())
        axis.set(title=f'{strength.capitalize()} opponent interaction', xlabel='Generation')
        axis.grid(alpha=0.2)
    axes[0].set_ylabel('Mean pure best-response gap (lower is better)')
    axes[0].legend()
    fig.savefig(output / 'progress.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11, 7), layout='constrained')
    limit = max(abs(value) for _, _, _, matrix in matrices for row in matrix for value in row) or 1
    for axis, (strength, method, generations, matrix) in zip(axes.flat, matrices):
        heatmap = axis.imshow(matrix, origin='lower', cmap='RdBu', vmin=-limit, vmax=limit)
        axis.set(title=f'{strength.capitalize()} / {method}', xlabel='B champion generation',
                 ylabel='A champion generation')
        ticks = sorted({0, len(generations) // 2, len(generations) - 1})
        labels = [generations[i] for i in ticks]
        axis.set_xticks(ticks, labels)
        axis.set_yticks(ticks, labels)
    fig.colorbar(heatmap, ax=axes, label='Payoff to A')
    fig.suptitle('Historical matchups on the first landscape')
    fig.savefig(output / 'historical_matchups.png', dpi=160)
    plt.close(fig)


def experiment(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('Output directory must be empty')
    seeds = random.Random(args.seed)
    pairs = [(seeds.getrandbits(64), seeds.getrandbits(64)) for _ in range(args.repeats)]
    metadata = {
        'bits': args.bits, 'generations': args.generations, 'checkpoint': args.checkpoint,
        'seed': args.seed, 'strengths': STRENGTHS, 'methods': METHODS,
        'candidates_per_side': POPULATION_SIZE, 'opponents_per_candidate': PANEL_SIZE,
        'training_matches_per_run': 2 * POPULATION_SIZE * PANEL_SIZE * args.generations,
        'seeds': [{'landscape': landscape, 'optimizer': optimizer} for landscape, optimizer in pairs],
    }
    (args.output / 'experiment.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    rows = []
    historical_rows = []
    matrices = []
    for repeat, (landscape_seed, optimizer_seed) in enumerate(pairs):
        game = Landscape(args.bits, landscape_seed)
        for name, strength in STRENGTHS.items():
            for method in METHODS:
                trace = run(game, strength, args.bits, optimizer_seed, method, args.generations, args.checkpoint)
                rows.extend({'repeat': repeat, 'strength': name, 'method': method, **row} for row in trace)
                matrix = []
                for a in trace:
                    matrix_row = []
                    for b in trace:
                        payoff = game.payoff(a['champion_a'], b['champion_b'], strength)
                        matrix_row.append(payoff)
                        historical_rows.append({
                            'repeat': repeat, 'strength': name, 'method': method,
                            'generation_a': a['generation'], 'generation_b': b['generation'], 'payoff_a': payoff,
                        })
                    matrix.append(matrix_row)
                if repeat == 0:
                    matrices.append((name, method, [row['generation'] for row in trace], matrix))
                print(f'Landscape {repeat + 1}/{args.repeats} | {name} | {method} | '
                      f'final gap {trace[-1]["pure_best_response_gap"]:.4f}', flush=True)

    for name, records in [('history.csv', rows), ('historical_matchups.csv', historical_rows)]:
        with (args.output / name).open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    summary = {}
    for strength in STRENGTHS:
        final = {
            method: [row['pure_best_response_gap'] for row in rows if row['strength'] == strength
                     and row['method'] == method and row['generation'] == args.generations]
            for method in METHODS
        }
        summary[strength] = {
            'mean_final_gap': {method: statistics.mean(gaps) for method, gaps in final.items()},
            'mean_paired_archive_minus_current': statistics.mean(
                a - b for a, b in zip(final['archive'], final['current'])
            ),
            'archive_better_than_current': sum(a < b for a, b in zip(final['archive'], final['current'])),
        }
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if not args.no_plots:
        save_plots(args.output, rows, matrices)
    print(f'Saved experiment to {args.output.resolve()}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bits', type=int, choices=range(3, 13), default=8)
    parser.add_argument('--generations', type=int, default=40)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--checkpoint', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=Path('results/coevolve-bits'))
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    if min(args.generations, args.repeats, args.checkpoint) < 1:
        parser.error('Generations, repeats, and checkpoint interval must be positive')
    try:
        experiment(args)
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
