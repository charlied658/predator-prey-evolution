import unittest
import numpy as np
from src.simulation.hunger_tree import (DEFAULTS,BASELINE,PREDATOR_BASELINE,PREDATOR_FEATURES,
    VisitMemory,action,random_tree,mutate,decide,episode,observe_predator)
from src.simulation.vision_pilot import setup
from src.experiments.hunger_coevolution import predator_score


class CoevolutionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.world=setup(dict(resolution=96,solid=[0]*96**2))

    def test_predator_baseline_tree_reproduces_original_fixed_controller(self):
        for scenario in (90000000,90000003,90000005):
            self.assertEqual(episode(BASELINE,scenario,self.world),episode(BASELINE,scenario,self.world,predator_tree=PREDATOR_BASELINE))

    def test_predator_receives_no_unseen_prey_or_food_information(self):
        p=np.array([48.,48.]);memory=VisitMemory(self.world);memory.mark(p,0)
        args=(np.zeros(2),np.array([10.,-10.]),memory,0,0,0,p,DEFAULTS)
        a=observe_predator(self.world,p,np.array([3.,3.]),*args)
        b=observe_predator(self.world,p,np.array([90.,90.]),*args)
        self.assertEqual(a,b);self.assertFalse(a['objects']);self.assertNotIn('hunger',a['features']);self.assertNotIn('food_count',a['features'])

    def test_starvation_is_not_rewarded_as_a_predator_capture(self):
        self.assertEqual(predator_score(dict(death='starved',survival=1),30),0)
        self.assertEqual(predator_score(dict(death='timeout',survival=30),30),0)
        self.assertGreater(predator_score(dict(death='captured',survival=5),30),predator_score(dict(death='captured',survival=20),30))

    def test_predator_mutations_have_role_appropriate_features_and_actions(self):
        rng=np.random.default_rng(713);tree=random_tree(rng,3,'predator')
        def check(node,depth=0):
            self.assertLessEqual(depth,3)
            if 'action' in node:
                if node['action'] in ('approach','away'):self.assertEqual(node['object'],'prey')
            else:
                self.assertIn(node['feature'],PREDATOR_FEATURES);check(node['yes'],depth+1);check(node['no'],depth+1)
        for _ in range(100):tree=mutate(tree,rng,3,'predator');check(tree)


if __name__=='__main__':unittest.main()
