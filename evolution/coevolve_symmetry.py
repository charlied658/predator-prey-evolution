"""Measure divergence and lead reversals in symmetric bit-string games."""
import argparse
import csv
import json
import random
import statistics
from pathlib import Path

from coevolve_bits import STRENGTHS, Landscape, run


class SymmetricLandscape(Landscape):
    def payoff(self, a, b, strength):
        return (super().payoff(a, b, strength) - super().payoff(b, a, strength)) / 2


def measure_trace(game, strength, trace):
    previous_leader = None
    for row in trace:
        a, b = row['champion_a'], row['champion_b']
        payoff = row['contemporary_payoff_a']
        leader = 'tie' if abs(payoff) <= 1e-12 else ('A' if payoff > 0 else 'B')
        reversed_lead = leader != 'tie' and previous_leader not in (None, leader)
        if leader != 'tie':
            previous_leader = leader
        # A common panel of all genomes measures strategy differences without
        # changing the opponents between A and B or between generations.
        profile_gap = statistics.mean(
            abs(game.payoff(a, rival, strength) - game.payoff(b, rival, strength))
            for rival in range(game.size)
        )
        yield {
            **row, 'hamming_distance': (a ^ b).bit_count(),
            'matchup_profile_gap': profile_gap,
            'leader': leader, 'lead_reversal': int(reversed_lead),
        }


def summarize(rows):
    rounds = rows[1:]
    leads = {side: sum(row['leader'] == side for row in rounds) for side in ('A', 'B', 'tie')}
    longest_lead = 0
    streak = 0
    previous = 'tie'
    for row in rounds:
        leader = row['leader']
        streak = 0 if leader == 'tie' else (streak + 1 if leader == previous else 1)
        longest_lead = max(longest_lead, streak)
        previous = leader
    return {
        'first_divergence': next((row['generation'] for row in rounds if row['hamming_distance']), None),
        'lead_reversals': sum(row['lead_reversal'] for row in rounds),
        'lead_generations': leads,
        'longest_unbroken_lead': longest_lead,
        'final_leader': rounds[-1]['leader'],
        'final_hamming_distance': rounds[-1]['hamming_distance'],
        'final_matchup_profile_gap': rounds[-1]['matchup_profile_gap'],
    }


def plot(output, examples):
    from matplotlib import use
    use('Agg')
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey='row', layout='constrained')
    for (name, rows), (payoff, divergence) in zip(examples.items(), axes.T):
        generations = [row['generation'] for row in rows]
        payoff.plot(generations, [row['contemporary_payoff_a'] for row in rows], color='#2369a1')
        payoff.axhline(0, color='gray', linewidth=0.8)
        payoff.set_title(f'{name.capitalize()} interaction')
        divergence.plot(generations, [row['hamming_distance'] for row in rows], color='#98531e')
        divergence.set_xlabel('Generation')
    axes[0, 0].set_ylabel('Matchup payoff to A')
    axes[1, 0].set_ylabel('Number of different bits')
    fig.suptitle('Lead reversals and genome divergence on the first landscape')
    fig.savefig(output / 'symmetry.png', dpi=160)
    plt.close(fig)


def experiment(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('Output directory must be empty')
    rng = random.Random(args.seed)
    traces, summaries, examples = [], [], {}
    for repeat in range(args.repeats):
        landscape_seed, optimizer_seed = rng.getrandbits(64), rng.getrandbits(64)
        game = SymmetricLandscape(8, landscape_seed)
        for name, strength in STRENGTHS.items():
            trace = run(game, strength, 8, optimizer_seed, 'current', args.generations, 1,
                        identical_start=True)
            rows = list(measure_trace(game, strength, trace))
            traces.extend({'repeat': repeat, 'strength': name, **row} for row in rows)
            summaries.append({'repeat': repeat, 'strength': name, 'landscape_seed': landscape_seed,
                              'optimizer_seed': optimizer_seed, **summarize(rows)})
            if repeat == 0:
                examples[name] = rows
        print(f'Landscape {repeat + 1}/{args.repeats}', flush=True)

    with (args.output / 'history.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(traces[0]))
        writer.writeheader()
        writer.writerows(traces)
    aggregate = {}
    for name in STRENGTHS:
        group = [row for row in summaries if row['strength'] == name]
        aggregate[name] = {
            'runs_diverged': sum(row['first_divergence'] is not None for row in group),
            'runs_with_reversals': sum(row['lead_reversals'] > 0 for row in group),
            'median_reversals': statistics.median(row['lead_reversals'] for row in group),
            'median_longest_lead': statistics.median(row['longest_unbroken_lead'] for row in group),
            'final_leaders': {side: sum(row['final_leader'] == side for row in group)
                              for side in ('A', 'B', 'tie')},
        }
    summary = {'seed': args.seed, 'generations': args.generations, 'strengths': STRENGTHS,
               'aggregate': aggregate, 'runs': summaries}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if not args.no_plots:
        plot(args.output, examples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generations', type=int, default=60)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=Path('results/coevolve-symmetry'))
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    if min(args.generations, args.repeats) < 1:
        parser.error('Generations and repeats must be positive')
    try:
        experiment(args)
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
