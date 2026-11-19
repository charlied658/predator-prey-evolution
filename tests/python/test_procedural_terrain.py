import unittest
import numpy as np
from src.terrain.procedural_terrain import generate_terrain
from src.simulation.vision_pilot import setup


class GeneratorChecks(unittest.TestCase):
    def test_reproducible_and_knobs_change_geometry(self):
        a=generate_terrain(42,.7,.35,16)
        self.assertEqual(a,generate_terrain(42,.7,.35,16))
        self.assertNotEqual(a['solid'],generate_terrain(43,.7,.35,16)['solid'])
        self.assertNotEqual(a['solid'],generate_terrain(42,.7,.8,16)['solid'])
        self.assertNotEqual(a['solid'],generate_terrain(42,.7,.35,25)['solid'])
        self.assertGreater(generate_terrain(42,.9)['open_fraction'],generate_terrain(42,.5)['open_fraction'])
        for kind in ('Outcrops','Ridges','Clearings'):
            collider,sites,_=setup(generate_terrain(42,kind=kind))
            self.assertGreater(len(sites),32)
            self.assertTrue(all(collider.clear(p) for p in sites[::100]))


if __name__=='__main__':unittest.main()
