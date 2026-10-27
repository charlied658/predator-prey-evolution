import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import coevolution as co
from godot_interface import evaluate_batch


class CoevolutionTests(unittest.TestCase):
    def test_safe_starts_and_balanced_archive_sampling(self):
        cases = co.scenarios(12, 100, co.OBSTACLES)
        self.assertEqual(cases, co.scenarios(12, 100, co.OBSTACLES))
        for case in cases:
            for name in ['predator_start', 'prey_start']:
                for obstacle in co.OBSTACLES:
                    self.assertGreater(math.dist(case[name], obstacle[:2]), obstacle[2] + 0.8)
        selected = co.select_opponents([[1], [2]], [{'genome': [9]}], 6, random.Random(1))
        self.assertTrue(all(x in [[1], [2]] for x in selected[:3]))
        self.assertEqual(selected[3:], [[9]] * 3)

    def test_collision_sensing_and_trace(self):
        right = [0.0] * 218
        right[204] = 10
        base = {'predator_genome': right, 'prey_controller': 'stationary',
                'predator_start': [-5, 0], 'prey_start': [8, 0],
                'predator_sensor_mode': 'ecology', 'record_trace': True}
        diagonal = right[:]
        diagonal[217] = 10
        sensor = [0.0] * 218
        sensor[7] = 2  # Ray pointing +x drives a hidden unit and then +x output.
        sensor[192] = 2
        cases = [dict(base, obstacles=[[0, 0, 2]]), dict(base, obstacles=[]),
                 dict(base, predator_genome=diagonal, predator_start=[-5, -1], obstacles=[[0, 0, 2]]),
                 dict(base, predator_genome=sensor, predator_start=[-3.5, 0], obstacles=[[0, 0, 2]]),
                 dict(base, predator_genome=sensor, predator_start=[-3.5, 0], obstacles=[])]
        rows = evaluate_batch(cases, 2)
        self.assertFalse(rows[0]['captured'])
        self.assertAlmostEqual(rows[0]['trajectory'][-1]['predator'][0], -2.5, places=3)
        self.assertGreater(rows[0]['predator_blocked_time'], 18)
        self.assertTrue(rows[1]['captured'])
        self.assertGreater(rows[2]['predator_turn_angle'], 0.1)
        for row in [rows[0], rows[2]]:
            self.assertGreater(row['predator_obstacle_time'], 0)
            for sample in row['trajectory']:
                self.assertGreaterEqual(math.hypot(*sample['predator']), 2.4999)
        self.assertLess(rows[3]['trajectory'][1]['predator'][0], rows[4]['trajectory'][1]['predator'][0])
        self.assertEqual(rows, evaluate_batch(cases, 1))

    def test_both_populations_evolve_and_resume_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory) / 'full', seed=7, population=4,
                                   generations=2, episodes=2, reference_cases=1, arena='obstacles',
                                   workers=1, matrix_every=1, matrix_cases=1, resume=False)
            with patch.object(co, 'analyze'):
                co.train(args)
                expected = json.loads((args.output / 'checkpoint.json').read_text())
                self.assertEqual(len(expected['predator_archive']), 3)
                self.assertEqual(len(expected['prey_archive']), 3)
                initial_rng = random.Random(args.seed)
                initial_predators = [co.new_genome(initial_rng) for _ in range(args.population)]
                initial_prey = [co.new_genome(initial_rng) for _ in range(args.population)]
                self.assertNotEqual(initial_predators, expected['predators'])
                self.assertNotEqual(initial_prey, expected['prey'])
                args.output = Path(directory) / 'broken'
                original = co.evaluate_role
                calls = 0
                def interrupted(*values):
                    nonlocal calls
                    calls += 1
                    if calls == 3:
                        raise RuntimeError('interrupted')
                    return original(*values)
                with patch.object(co, 'evaluate_role', side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                        co.train(args)
                args.resume = True
                co.train(args)
                actual = json.loads((args.output / 'checkpoint.json').read_text())
                self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
