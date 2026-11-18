"""Regression checks for cell edges, rounded corners, and swept impacts."""
import random
import unittest

from grid_collision import GridCollider


class GridCollisionChecks(unittest.TestCase):
    def setUp(self):
        n = 32
        solid = [int(x==16) for y in range(n) for x in range(n)]
        self.collider = GridCollider({'resolution':n, 'solid':solid}, radius=.7)

    def test_high_speed_impact_does_not_tunnel(self):
        point, contacts = self.collider.move((5, 10), (50, 0))
        self.assertGreater(contacts, 0)
        self.assertAlmostEqual(point[0], 15.3, places=4)

    def test_slide_and_repeated_pressure(self):
        point, _ = self.collider.move((10, 10), (12, 5))
        self.assertAlmostEqual(point[1], 15, places=4)
        for _ in range(100):
            point, _ = self.collider.move(point, (.1, .05))
        self.assertTrue(self.collider.clear(point))
        self.assertAlmostEqual(point[1], 20, places=3)

    def test_rounded_corner_and_arena_boundary(self):
        solid = [0]*32**2
        solid[16*32+16] = 1
        c = GridCollider({'resolution':32, 'solid':solid}, radius=.7)
        point, contacts = c.move((13, 13), (6, 6))
        self.assertGreater(contacts, 0)
        self.assertTrue(c.clear(point))
        point, _ = c.move((5, 5), (-100, -50))
        self.assertTrue(c.clear(point))

    def test_random_stress(self):
        rng = random.Random(42)
        point = (8, 8)
        for _ in range(1000):
            point, _ = self.collider.move(point, (rng.uniform(-50,50), rng.uniform(-50,50)))
            self.assertTrue(self.collider.clear(point))


if __name__ == '__main__':
    unittest.main()
