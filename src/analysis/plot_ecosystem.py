"""Plot saved numeric comparisons without rerunning the ecosystem."""
import argparse
import csv
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

LABELS = {'es': 'Mutation-only ES', 'ga': 'Genetic algorithm', 'de': 'Differential evolution', 'random': 'Random search'}
COLORS = {'es': '#2166ac', 'ga': '#d97706', 'de': '#16806a', 'random': '#8b5ca6'}


def make_plots(output):
    os.environ.setdefault('MPLCONFIGDIR', str(output / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    metadata = json.loads((output / 'experiment.json').read_text())
    with (output / 'history.csv').open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError('No history to plot')
    by_method = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for field in ('generation', 'candidate_evaluations', 'run'):
            row[field] = int(row[field])
        for field in ('training_fitness', 'validation_fitness', 'wall_seconds',
                      'baseline_training_fitness', 'baseline_validation_fitness'):
            row[field] = float(row[field])
        by_method[row['algorithm']][row['run']].append(row)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'figure.facecolor': 'white'})

    def panel(axis, x_field, y_field, title, xlabel, baseline_field):
        for method in metadata['algorithms']:
            runs = by_method.get(method, {})
            if not runs:
                continue
            grouped = defaultdict(list)
            for trace in runs.values():
                trace.sort(key=lambda r: r['generation'])
                if len(runs) > 1:
                    axis.plot([r[x_field] for r in trace], [r[y_field] for r in trace],
                              color=COLORS[method], alpha=0.22, linewidth=1)
                for row in trace:
                    grouped[row['generation']].append(row)
            generations = sorted(grouped)
            x = [statistics.mean(row[x_field] for row in grouped[g]) for g in generations]
            y = [statistics.mean(row[y_field] for row in grouped[g]) for g in generations]
            lower = [min(row[y_field] for row in grouped[g]) for g in generations]
            upper = [max(row[y_field] for row in grouped[g]) for g in generations]
            axis.plot(x, y, '-o', color=COLORS[method], label=LABELS[method], linewidth=2, markersize=4)
            if len(runs) > 1:
                axis.fill_between(x, lower, upper, color=COLORS[method], alpha=0.10)
        baseline = statistics.mean(row[baseline_field] for row in rows)
        axis.axhline(baseline, color='#68717c', linestyle='--', linewidth=1.2, label='Original rules')
        axis.set(title=title, xlabel=xlabel, ylabel='Fitness (higher is better)')
        axis.grid(axis='y', alpha=0.18)
        if x_field != 'wall_seconds':
            axis.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        axis.legend(frameon=False, fontsize=9)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9))
    panel(axes[0], 'generation', 'training_fitness', 'Best fitness on training scenarios', 'Generation', 'baseline_training_fitness')
    panel(axes[1], 'generation', 'validation_fitness', 'Held-out scenarios (not used for selection)', 'Generation', 'baseline_validation_fitness')
    fig.suptitle(f'Ecosystem evolution: {metadata["role"]} behavior', fontsize=15, fontweight='bold')
    note = (f'{metadata["runs"]} paired optimizer run(s); equal candidate budgets and shared initial populations. '
            + ('Lines show means; shading is observed min–max, not a confidence interval.'
               if metadata['runs'] > 1 else 'One optimizer seed: exploratory curves, not evidence of superiority.'))
    fig.text(0.5, 0.025, note, ha='center', fontsize=8.5, color='#505963')
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    for extension in ('png', 'svg'):
        fig.savefig(output / f'fitness.{extension}', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9))
    panel(axes[0], 'candidate_evaluations', 'training_fitness', 'Progress per candidate budget', 'Candidate evaluation slots', 'baseline_training_fitness')
    panel(axes[1], 'wall_seconds', 'training_fitness', 'Progress per computation time', 'Elapsed wall time per run (seconds)', 'baseline_training_fitness')
    fig.suptitle('Fitness versus work', fontsize=15, fontweight='bold')
    fig.text(0.5, 0.025, 'Elapsed time includes training and validation; repeated genomes are cached. Timing depends on this machine and execution order.',
             ha='center', fontsize=8.5, color='#505963')
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    for extension in ('png', 'svg'):
        fig.savefig(output / f'fitness_budget.{extension}', dpi=160)
    plt.close(fig)
    print(f'Plots: {output / "fitness.png"} and {output / "fitness_budget.png"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='Existing comparison result folder')
    args = parser.parse_args()
    try:
        make_plots(args.output.resolve())
    except ImportError:
        parser.exit(1, 'Install requirements.txt to plot, then rerun this command; no simulations need repeating.\n')
    except (ValueError, OSError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
