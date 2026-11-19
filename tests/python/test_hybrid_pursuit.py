"""Checks for priority, information boundaries, recovery and physical capture."""
import unittest
from unittest.mock import patch
import numpy as np
from src.training.hybrid_pursuit import OrganismPolicy, RULES, capture_fraction, episodes
from src.simulation.vision_pilot import setup, GENES, DT
from src.terrain.grid_collision import GridCollider


class PursuitChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = setup(dict(resolution=96, solid=[0]*96**2))

    def policy(self, role):
        return OrganismPolicy(role, np.array([[48., 48.]]), self.world, RULES)

    def command(self, policy, distance, tick=0):
        return policy.command(np.zeros((1, GENES)), np.array([[48., 48.]]),
                              np.array([[48.+distance, 48.]]), np.zeros((1,18)),
                              np.array([[[48.,56.]]]), np.ones((1,1),bool),
                              tick, np.ones(1,bool))

    def test_predator_pursues_only_visible_prey(self):
        p = self.policy(0)
        self.command(p,20)
        self.assertEqual(p.modes[0], 'pursue')
        np.testing.assert_allclose(p.goals, [[68,48]])
        self.command(p,30)
        self.assertEqual(p.modes[0], 'explore')

    def test_prey_priority_and_hysteresis(self):
        p = self.policy(1)
        self.assertLess(self.command(p,26)[0,0],0)
        self.assertEqual(p.modes[0], 'avoid')
        self.command(p,28)
        self.assertEqual(p.modes[0], 'avoid')
        self.command(p,31)
        self.assertEqual(p.modes[0], 'forage')
        self.assertTrue(p.seen[0])

    def test_danger_interrupts_food_recovery(self):
        p = self.policy(1)
        p.until[0]=100
        p.escape[0]=[60,48]
        command=self.command(p,20)
        self.assertEqual(p.modes[0],'avoid')
        self.assertLess(command[0,0],0)
        self.assertEqual(p.food_cooldown[0,0],0)

    def test_stuck_avoidance_does_not_back_toward_predator(self):
        p=self.policy(1)
        command=self.command(p,20,p.window)
        self.assertEqual(p.modes[0],'recover-avoid')
        self.assertLessEqual(command[0,0],0)
        self.assertGreaterEqual(np.linalg.norm(command[0]),RULES['speed']*.8-1e-9)
        self.assertEqual(p.food_cooldown[0,0],0)

    def test_unseen_opponent_does_not_change_network_inputs(self):
        for role in (0,1):
            inputs=[]
            with patch('src.training.hybrid_pursuit.network', side_effect=lambda g,x: inputs.append(x.copy()) or np.zeros((1,2))):
                self.command(self.policy(role),40)
                self.command(self.policy(role),44)
            np.testing.assert_array_equal(*inputs)

    def test_visit_maps_are_separate(self):
        a,b=self.policy(0),self.policy(1)
        a.mark(np.array([[80.,80.]]),1,np.ones(1,bool))
        self.assertEqual(a.visited.sum(),2)
        self.assertEqual(b.visited.sum(),1)

    def test_continuous_capture_and_near_miss(self):
        a=[(0,(0,0)),(1,(10,0))]
        b=[(0,(10,0)),(1,(0,0))]
        self.assertAlmostEqual(capture_fraction(a,b,2),.4)
        self.assertIsNone(capture_fraction(a,[(0,(10,3)),(1,(0,3))],2))
        # Piecewise paths differ from the straight chord between endpoints.
        bent=[(0,(0,0)),(.5,(0,10)),(1,(10,10))]
        self.assertIsNone(capture_fraction(bent,[(0,(5,5)),(1,(5,5))],1))

    def test_collision_path_preserves_solver_and_time(self):
        c=GridCollider(dict(resolution=32,solid=[int(x==16) for y in range(32) for x in range(32)]),radius=.7)
        path=[]
        traced=c.move((10,10),(12,5),path=path)
        self.assertEqual(traced,c.move((10,10),(12,5)))
        self.assertEqual(path[0][0],0)
        self.assertEqual(path[-1][0],1)
        self.assertTrue(all(b[0]>a[0] for a,b in zip(path,path[1:])))
        self.assertTrue(all(c.clear(p) for _,p in path))

    def test_capture_ends_encounter_without_respawn(self):
        positions=np.array([[[40.,48.],[45.,48.]]])
        pellets=np.array([[[45.,48.]]])
        def toward(policy,*args,**kwargs):
            return np.array([[18. if policy.role==0 else -18.,0.]])
        trace=[]
        with patch('src.training.hybrid_pursuit.make_arrangements',return_value=(positions,pellets)), patch.object(OrganismPolicy,'command',toward):
            result=episodes(np.zeros((1,GENES)),np.zeros((1,GENES)),[0],self.world,RULES,seconds=1,pellet_count=1,trace=trace)
        self.assertEqual(result[0,2],1)
        self.assertAlmostEqual(result[0,3],(5-4.8)/36)
        self.assertEqual(result[0,4],0)
        self.assertEqual(len(trace),2)
        self.assertTrue(trace[-1]['caught'])


if __name__=='__main__':
    unittest.main()
