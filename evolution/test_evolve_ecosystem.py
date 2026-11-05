"""Focused checks for the minimal ecosystem evolutionary loop."""
import argparse
import json
import random
import subprocess
import tempfile
import unittest
from pathlib import Path

import evolve_ecosystem as evolution
from config import GODOT_PATH, PROJECT_PATH


class BasicEvolutionChecks(unittest.TestCase):
    def test_mutation_is_reproducible_bounded_and_does_not_edit_parent(self):
        parent = evolution.DEFAULTS['prey'].copy()
        child = evolution.mutate(parent, random.Random(42))
        self.assertEqual(child, evolution.mutate(parent, random.Random(42)))
        self.assertNotEqual(child, parent)
        self.assertEqual(parent, evolution.DEFAULTS['prey'])
        for key, value in child.items():
            low, high = evolution.BOUNDS[key]
            self.assertTrue(low <= value <= high)

    def test_scores_reward_food_capture_and_penalize_death(self):
        row = {'captures': 3, 'food_eaten': 8, 'starvations': {'predator': 1, 'prey': 2}}
        self.assertEqual(evolution.fitness(row, 'predator'), 1)
        self.assertEqual(evolution.fitness(row, 'prey'), -2)

    def test_gene_changes_affect_the_godot_controller(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([GODOT_PATH, '--headless', '--log-file', str(Path(directory) / 'check.log'),
                                     '--path', PROJECT_PATH, '--script', 'res://tests/ecosystem_basic_evolution.gd'],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn('SCRIPT ERROR', result.stdout + result.stderr)
            self.assertIn('Basic evolution controller check: PASS', result.stdout)

    def test_real_two_generation_run_keeps_elite_and_replays_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            args = argparse.Namespace(config=Path(__file__).with_name('ecosystem.json'), role='prey',
                                      population=3, generations=2, duration=2, seed=42, mutation=0.15, output=output)
            history = evolution.train(args)
            scores = [row['fitness'] for row in history]
            self.assertEqual(scores, sorted(scores))
            timing = json.loads((output / 'timing.json').read_text())
            self.assertLessEqual(timing['episodes_run'], 5)  # 1 original + 2 new copies per generation.
            original = json.loads((output / 'baseline.json').read_text())
            best = json.loads((output / 'best.json').read_text())
            self.assertEqual(original['seed'], best['seed'])
            self.assertNotIn('predator', best['behavior'])
            expected = [json.loads(path.read_text()) for path in (output / 'episodes').glob('*.summary.json')]
            expected = next(row for row in expected if row['settings']['behavior'] == best['behavior'])
            replay = evolution.run_episode(best, output, 999)
            for key in ['captures', 'food_eaten', 'starvations', 'mean_energy']:
                self.assertEqual(expected[key], replay[key])
            with self.assertRaisesRegex(ValueError, 'not empty'):
                evolution.train(args)


if __name__ == '__main__':
    unittest.main()
