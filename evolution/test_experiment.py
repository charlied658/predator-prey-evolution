"""Run with: python -m unittest discover -s evolution -p 'test_*.py'"""
import math
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiment import bootstrap, breed, evaluate, fitness, metrics, pursuit_genome, scenario, suite, train
from godot_interface import evaluate_batch
from replay import resolve_controller


class ExperimentTests(unittest.TestCase):
    def test_seeded_scenarios_and_suites(self):
        self.assertEqual(suite(42, 3), suite(42, 3))
        self.assertNotEqual(suite(42, 3), suite(43, 3))
        rng = random.Random(19)
        for _ in range(100):
            case = scenario(rng)
            self.assertTrue(5 <= math.dist(case['predator_start'], case['prey_start']) <= 20)
            self.assertTrue(all(abs(x) <= 12 for p in case.values() for x in p))

    def test_shaping_only_changes_failures(self):
        capture = {'captured': True, 'predator_fitness': 110, 'minimum_distance': 0.9}
        miss = {'captured': False, 'predator_fitness': 0, 'minimum_distance': 2}
        self.assertEqual(fitness(capture, True), 110)
        self.assertEqual(fitness(miss, False), 0)
        self.assertAlmostEqual(fitness(miss, True), 1 / 3)
        self.assertIsNone(bootstrap([0.5]))
        self.assertEqual(bootstrap([1, 1, 1]), [1, 1])

    def test_pursuit_reward_rejects_lucky_pass_and_wall_pushing(self):
        def row(mean, alignment):
            return {'captured': False, 'predator_fitness': 0, 'minimum_distance': 0.1,
                    'initial_distance': 10, 'mean_distance': mean, 'approach_alignment': alignment}
        self.assertGreater(fitness(row(3, 0.8), False, True), fitness(row(12, 0), False, True))
        self.assertEqual(fitness(row(10, 0), False, True), 0)
        self.assertLess(fitness(row(15, -0.5), False, True), 0)

    def test_elites_preserved_without_aliasing(self):
        population = [[float(i)] * 66 for i in range(5)]
        result = breed(population, [0, 1, 2, 3, 4], random.Random(2))
        self.assertEqual(result[:2], [population[4], population[3]])
        result[0][0] = 999
        self.assertEqual(population[4][0], 4)

    def test_pursuit_initialization_and_explicit_replay_selection(self):
        weights = pursuit_genome(random.Random(1))
        self.assertEqual(len(weights), 66)
        for i in range(8):
            self.assertEqual(weights[i * 6 + 4:i * 6 + 6], [0, 0])
        self.assertEqual([weights[56], weights[65]], [0, 0])
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed, validation in [(1, 0.25), (2, 0.75)]:
                folder = root / f'diverse-seed-{seed}'
                folder.mkdir()
                (folder / 'best_predator.json').write_text('{}')
                (folder / 'report.json').write_text(json.dumps({'history': [{'champion_validation_capture_rate': validation}]}))
            self.assertEqual(resolve_controller(root, 'diverse', None).parent.name, 'diverse-seed-2')
            self.assertEqual(resolve_controller(root, 'diverse', 1).parent.name, 'diverse-seed-1')
            with self.assertRaises(ValueError):
                resolve_controller(root, 'fixed', None)


class GodotIntegrationTests(unittest.TestCase):
    def test_interrupted_resume_matches_uninterrupted_training(self):
        for mode in ['legacy', 'pursuit']:
            with self.subTest(mode=mode):
                self.check_resume(mode)

    def check_resume(self, mode):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory) / 'full', population=3,
                                   pool_size=2, resume=False, generations=2,
                                   episodes=1, workers=1, shaping=False, training_mode=mode)
            validation, test = suite(123, 1), suite(456, 1)
            expected = train(args, 'diverse', 37, validation, test)
            args.output = Path(directory) / 'interrupted'
            calls = 0
            def interrupt_after_checkpoint(*values):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError('simulated interruption')
                return evaluate(*values)
            with patch('experiment.evaluate', side_effect=interrupt_after_checkpoint):
                with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    train(args, 'diverse', 37, validation, test)
            args.resume = True
            actual = train(args, 'diverse', 37, validation, test)
            self.assertEqual(expected, actual)
            for name in ['checkpoint.json', 'best_predator.json', 'test_encounters.json']:
                self.assertEqual((Path(directory) / 'full' / 'diverse-seed-37' / name).read_text(),
                                 (args.output / 'diverse-seed-37' / name).read_text())

    def test_known_trajectories_and_worker_independence(self):
        zero = [0.0] * 66
        right = zero.copy()
        right[56] = 10.0  # First output bias: deterministic +x motion.
        cases = [
            {'predator_genome': zero, 'prey_controller': 'stationary',
             'predator_start': [-5, 0], 'prey_start': [5, 0]},
            {'predator_genome': right, 'prey_controller': 'stationary',
             'predator_start': [-5, 0], 'prey_start': [5, 0]},
            {'predator_genome': zero, 'prey_controller': 'straight',
             'predator_start': [-5, 0], 'prey_start': [5, 0]},
            {'predator_genome': zero, 'prey_controller': 'flee',
             'predator_start': [-5, 0], 'prey_start': [5, 0]},
        ]
        serial = evaluate_batch(cases)
        parallel = evaluate_batch(cases, workers=2)
        self.assertEqual(serial, parallel)
        self.assertFalse(serial[0]['captured'])
        self.assertAlmostEqual(serial[0]['time'], 20, places=4)
        self.assertEqual(serial[0]['minimum_distance'], 10)
        self.assertAlmostEqual(serial[0]['mean_distance'], 10, places=4)
        self.assertEqual(serial[0]['approach_alignment'], 0)
        self.assertTrue(serial[1]['captured'])
        self.assertAlmostEqual(serial[1]['time'], 3, delta=1 / 60 + 0.001)
        self.assertAlmostEqual(serial[1]['predator_path_length'], 9, delta=0.06)
        self.assertAlmostEqual(serial[1]['approach_alignment'], 1, places=4)
        self.assertGreater(serial[2]['prey_boundary_time'], 15)
        self.assertGreater(serial[3]['prey_boundary_time'], 16)
        self.assertEqual(metrics([serial[0]])['capture_time'], None)

    def test_bad_genome_fails_without_stale_results(self):
        with self.assertRaisesRegex(RuntimeError, 'Godot failed'):
            evaluate_batch([{'predator_genome': [], 'prey_controller': 'stationary'}])


if __name__ == '__main__':
    unittest.main()
