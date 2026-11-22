import unittest
import numpy as np
from src.experiments.predator_fitness_pilot import fitness,distance_progress,FOOD_ONLY_PREY
from src.simulation.hunger_tree import BASELINE,PREDATOR_BASELINE,DEFAULTS,episode,decide
from src.simulation.vision_pilot import setup


class PredatorFitnessChecks(unittest.TestCase):
    def test_food_only_prey_ignores_predator_and_explores_without_food(self):
        for distance in (0.,10.,90.):
            features=dict(food_count=1.,predator_distance=distance,predator_visible=1.)
            self.assertEqual(decide(FOOD_ONLY_PREY,features)['object'],'food')
            features['food_count']=0.
            self.assertEqual(decide(FOOD_ONLY_PREY,features)['selector'],'unvisited')
        self.assertNotEqual(FOOD_ONLY_PREY,BASELINE)

    def test_progress_is_bounded_and_capture_stays_distinct(self):
        no_progress=dict(initial_distance=30.,closest_distance=30.)
        progress=dict(initial_distance=30.,closest_distance=15.)
        self.assertEqual(distance_progress(no_progress),0)
        self.assertEqual(distance_progress(progress),.5)
        for death in ('starved','timeout'):
            r=dict(death=death,survival=10.)
            self.assertEqual(fitness(r,progress,30,0),0)
            self.assertAlmostEqual(fitness(r,progress,30,.1),.05)
            self.assertLessEqual(fitness(r,progress,30,.5),.5)
        self.assertGreaterEqual(fitness(dict(death='captured',survival=30.),no_progress,30,.5),1)

    def test_diagnostics_preserve_episode_and_match_recorded_distances(self):
        world=setup(dict(resolution=96,solid=[0]*96**2));config=dict(DEFAULTS,seconds=3)
        first=episode(BASELINE,350000000,world,config,predator_tree=PREDATOR_BASELINE)
        diagnostics={};trace=[]
        measured=episode(BASELINE,350000000,world,config,trace,predator_tree=PREDATOR_BASELINE,diagnostics=diagnostics)
        self.assertEqual(first,measured)
        distances=[float(np.linalg.norm(np.array(f['predator'])-f['prey'])) for f in trace]
        self.assertAlmostEqual(diagnostics['initial_distance'],distances[0])
        self.assertAlmostEqual(diagnostics['closest_distance'],min(distances))
        self.assertAlmostEqual(diagnostics['final_distance'],distances[-1])


if __name__=='__main__':unittest.main()
