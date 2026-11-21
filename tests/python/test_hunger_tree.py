import unittest
import numpy as np
from src.simulation.hunger_tree import (DEFAULTS,FEATURES,BASELINE,VisitMemory,action,decide,random_tree,mutate,
                                       hunger_events,observe,episode,episode_steps,viewer_steps,destination)
from src.simulation.vision_pilot import setup


class HungerTreeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.world=setup(dict(resolution=96,solid=[0]*96**2))

    def test_starvation_and_food_event_order(self):
        config=dict(DEFAULTS,maximum_hunger=10,pellet_energy=3,metabolism=1)
        t,death,energy,food=hunger_events(.04,[(.06,'food',0)],.1,config)
        self.assertAlmostEqual(t,.04);self.assertEqual(death,'starved');self.assertFalse(food)
        t,death,energy,food=hunger_events(.04,[(.02,'food',0)],.1,config)
        self.assertIsNone(death);self.assertEqual(food,[0]);self.assertAlmostEqual(energy,2.94)

    def test_capture_wins_food_tie_and_reserve_is_capped(self):
        config=dict(DEFAULTS,maximum_hunger=10,pellet_energy=6)
        t,death,energy,food=hunger_events(9,[(.03,'food',0),(.03,'capture',-1)],.1,config)
        self.assertEqual(death,'captured');self.assertFalse(food);self.assertAlmostEqual(t,.03)
        _,death,energy,food=hunger_events(9,[(.03,'food',0)],.1,config)
        self.assertIsNone(death);self.assertEqual(food,[0]);self.assertAlmostEqual(energy,9.93)

    def test_hidden_objects_do_not_enter_observation_or_target_selection(self):
        position=np.array([48.,48.]);memory=VisitMemory(self.world);memory.mark(position,0)
        pellets=np.array([[55.,48.],[90.,90.]])
        obs=observe(self.world,position,np.array([5.,5.]),pellets,np.ones(2,bool),10,np.zeros(2),memory,0,0,0,DEFAULTS)
        self.assertEqual(len(obs['objects']),1)
        self.assertEqual(obs['features']['predator_distance'],96)
        np.testing.assert_array_equal(destination(action('approach',object='predator',rank=0,order='distance'),obs,self.world,position,memory,0,np.random.default_rng(1),DEFAULTS),position)
        self.assertIn('route_distance',obs['objects'][0]);self.assertIn('waypoint',obs['objects'][0])

    def test_mutation_produces_valid_bounded_trees(self):
        rng=np.random.default_rng(23);tree=random_tree(rng,3)
        def depth(node):return 0 if 'action' in node else 1+max(depth(node['yes']),depth(node['no']))
        for _ in range(200):
            tree=mutate(tree,rng,3)
            self.assertLessEqual(depth(tree),3)
            leaf=decide(tree,{key:(lo+hi)/2 for key,(lo,hi) in FEATURES.items()})
            self.assertIn(leaf['action'],('stay','approach','away','location'))

    def test_stationary_dies_and_foraging_can_extend_life(self):
        config=dict(DEFAULTS,predator=False)
        stationary=episode(action('stay'),90000000,self.world,config)
        self.assertEqual(stationary['death'],'starved');self.assertEqual(stationary['food'],0)
        self.assertAlmostEqual(stationary['survival'],10)
        first=episode(BASELINE,90000000,self.world,config)
        self.assertGreater(first['survival'],stationary['survival']);self.assertGreater(first['food'],0)
        self.assertEqual(first,episode(BASELINE,90000000,self.world,config))

    def test_live_steps_and_viewer_adapter_preserve_scored_episode(self):
        config=dict(DEFAULTS,seconds=2,predator=False)
        expected=episode(BASELINE,90000000,self.world,config)
        steps=episode_steps(BASELINE,90000000,self.world,config,emit=True);raw=[]
        while True:
            try:raw.append(next(steps))
            except StopIteration as finished:actual=finished.value;break
        self.assertEqual(expected,actual)
        frames=list(viewer_steps(BASELINE,90000000,self.world,config))
        self.assertEqual(len(frames),len(raw))
        for frame,source in zip(frames,raw):
            self.assertEqual(frame['seconds'],source['seconds'])
            self.assertEqual(frame['positions'][1],source['prey'])
            self.assertEqual(frame['hunger'],source['hunger'])
        self.assertEqual(frames[-1]['death'],'timeout')


if __name__=='__main__':unittest.main()
