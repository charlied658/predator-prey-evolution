"""Budget fairness, operators, numeric/visual parity, and plot output."""
import argparse
import importlib.util
import json
import random
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import compare_ecosystem as comparison
from ecosystem_numeric import evaluate
from evolve_ecosystem import BOUNDS, DEFAULTS
from config import GODOT_PATH, PROJECT_PATH


class ComparisonChecks(unittest.TestCase):
    def args(self, output):
        return argparse.Namespace(config=Path(__file__).with_name('ecosystem.json'), role='prey',
                                  algorithms=['es', 'ga', 'de'], population=4, generations=2,
                                  runs=1, duration=0.25, seed=100, train_seeds=[42], validation_seeds=[10042],
                                  mutation=0.15, workers=1, max_episodes=60, no_plots=True, output=output)

    def test_operators_are_seeded_bounded_and_preserve_parents(self):
        population = comparison.initial_population(DEFAULTS['prey'], 4, 42, 0.15)
        before = json.dumps(population)
        for method in comparison.ALGORITHMS:
            children, targets = comparison.offspring(method, population, [1, 4, 2, 0], random.Random(3), 0.15)
            self.assertEqual((children, targets), comparison.offspring(method, population, [1, 4, 2, 0], random.Random(3), 0.15))
            self.assertEqual(len(children), 3)
            for child in children:
                for key, value in child.items():
                    self.assertTrue(BOUNDS[key][0] <= value <= BOUNDS[key][1])
            next_population, next_scores = comparison.select_population(method, population, [1, 4, 2, 0], children, [-1, -1, -1], targets)
            self.assertEqual(len(next_population), 4)
            self.assertEqual(max(next_scores), 4)
        self.assertEqual(json.dumps(population), before)

    def test_limits_and_seed_overlap_rejected_before_evaluation(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(comparison, 'evaluate') as evaluate_mock:
            args = self.args(Path(directory))
            args.max_episodes = 1
            with self.assertRaisesRegex(ValueError, 'episodes'):
                comparison.compare(args)
            args.max_episodes = 60
            args.validation_seeds = args.train_seeds
            with self.assertRaisesRegex(ValueError, 'overlap'):
                comparison.compare(args)
            evaluate_mock.assert_not_called()

    def test_numeric_batch_parallel_order_and_visual_parity(self):
        settings = json.loads(Path(__file__).with_name('ecosystem.json').read_text())
        jobs = [{**settings, 'duration': 0.5, 'seed': seed} for seed in [42, 43]]
        self.assertEqual(evaluate(jobs), evaluate(jobs, workers=2))
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([GODOT_PATH, '--headless', '--log-file', str(Path(directory) / 'test.log'),
                                     '--path', PROJECT_PATH, '--script', 'res://tests/ecosystem_numeric_equivalence.gd'],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn('SCRIPT ERROR', result.stdout + result.stderr)
            self.assertIn('Numeric/visual equivalence: PASS', result.stdout)

    def test_real_comparison_budgets_initial_population_cache_and_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'comparison'
            args = self.args(output)
            rows = comparison.compare(args)
            originals = []
            for method in args.algorithms:
                group = [row for row in rows if row['algorithm'] == method]
                self.assertEqual([row['candidate_evaluations'] for row in group], [4, 7, 10])
                scores = [row['training_fitness'] for row in group]
                self.assertEqual(scores, sorted(scores))
                folder = output / method / 'run-000'
                originals.append(json.loads((folder / 'population-0000.json').read_text()))
                self.assertTrue((folder / 'best.json').exists())
                self.assertEqual(json.loads((folder / 'validation.json').read_text())['seed'], 10042)
            self.assertEqual(originals[0], originals[1])
            self.assertEqual(originals[1], originals[2])
            summary = json.loads((output / 'summary.json').read_text())
            self.assertLessEqual(summary['actual_episodes'], summary['maximum_episodes'])
            evaluator = comparison.Evaluator(json.loads(args.config.read_text()), 'prey', [42], 1, output, 'cache-check')
            evaluator.base['duration'] = 0.25
            evaluator.scores([DEFAULTS['prey'], DEFAULTS['prey']])
            evaluator.scores([DEFAULTS['prey']])
            self.assertEqual(evaluator.episodes, 1)
            if importlib.util.find_spec('matplotlib'):
                from plot_ecosystem import make_plots
                make_plots(output)
                self.assertGreater((output / 'fitness.png').stat().st_size, 1000)
                self.assertTrue((output / 'fitness.svg').exists())
                self.assertTrue((output / 'fitness_budget.png').exists())


if __name__ == '__main__':
    unittest.main()
