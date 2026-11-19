import unittest
import numpy as np
from src.terrain.grid_collision import GridCollider
from src.training.hybrid_pursuit import OrganismPolicy,RULES
from src.simulation.prey_interactions import RandomPreyTargets,move_prey,pickup_times


class InteractionChecks(unittest.TestCase):
    def setUp(self):
        self.collider=GridCollider(dict(resolution=96,solid=[0]*96**2))

    def test_head_on_collision_does_not_tunnel(self):
        pos,paths,hits,_=move_prey(self.collider,[[30,48],[60,48]],[[40,0],[-40,0]],np.ones(2,bool))
        self.assertGreaterEqual(np.linalg.norm(pos[0]-pos[1]),4.8-1e-7)
        self.assertLess(pos[0,0],pos[1,0])
        self.assertTrue(np.isnan(hits).all())
        np.testing.assert_allclose(pos.mean(axis=0),[45,48],atol=1e-5)
        self.assertGreater(len(paths[0]),2)

    def test_glancing_contacts_keep_tangential_motion(self):
        pos,_,_,_=move_prey(self.collider,[[30,48],[40,51]],[[20,6],[0,0]],np.ones(2,bool))
        self.assertGreaterEqual(np.linalg.norm(pos[0]-pos[1]),4.8-1e-7)
        self.assertGreater(pos[0,1],48)
        self.assertGreater(np.linalg.norm(pos[1]-[40,51]),0)

    def test_group_cannot_push_a_body_into_a_wall(self):
        solid=[int(x==60) for y in range(96) for x in range(96)]
        c=GridCollider(dict(resolution=96,solid=solid))
        pos=np.array([[47.,48.],[52.,48.],[57.,48.]])
        for _ in range(15):
            pos,_,_,_=move_prey(c,pos,np.array([[2.,.2],[2.,.2],[2.,.2]]),np.ones(3,bool))
            self.assertTrue(all(c.clear(p) for p in pos))
            for i in range(3):
                for j in range(i):self.assertGreaterEqual(np.linalg.norm(pos[i]-pos[j]),4.8-1e-6)

    def test_capture_removes_body_before_later_peer_contact(self):
        pred=[(0,np.array([30.,48.])),(1,np.array([50.,48.]))]
        pos,_,hits,_=move_prey(self.collider,[[40,48],[50,48]],[[0,0],[-10,0]],np.ones(2,bool),pred)
        self.assertTrue(np.isfinite(hits).all())
        self.assertLess(hits[0],hits[1])

    def test_inactive_body_is_not_an_obstacle(self):
        pos,_,_,_=move_prey(self.collider,[[30,48],[40,48]],[[20,0],[0,0]],np.array([True,False]))
        np.testing.assert_allclose(pos[0],[50,48])

    def test_repeated_boundary_pressure_stays_clear(self):
        pos=np.array([[93.6,80.6],[89.28,78.5],[58.,54.]])
        velocity=np.array([[.763,-1.221],[.263,1.416],[-.498,.829]])
        for _ in range(20):
            pos,_,_,_=move_prey(self.collider,pos,velocity,np.ones(3,bool))
            self.assertTrue(all(self.collider.clear(p) for p in pos))

    def test_seeded_group_stress_preserves_separation(self):
        rng=np.random.default_rng(192)
        pos=np.array([[30.,30.],[36.,30.],[42.,30.],[36.,36.]])
        for _ in range(120):
            pos,_,_,_=move_prey(self.collider,pos,rng.uniform(-3,3,(4,2)),np.ones(4,bool))
            for i in range(4):
                self.assertTrue(self.collider.clear(pos[i]))
                for j in range(i):self.assertGreaterEqual(np.linalg.norm(pos[i]-pos[j]),4.8-1e-6)

    def test_piecewise_pickup_does_not_use_straight_chord(self):
        path=[(0,np.array([0.,0.])),(.5,np.array([0.,10.])),(1,np.array([10.,10.]))]
        times=pickup_times([path],np.array([[5.,5.],[0.,5.]]),np.array([True]),1)
        self.assertTrue(np.isinf(times[0,0]))
        self.assertAlmostEqual(times[0,1],.2)

    def test_independent_targets_are_persistent_and_reproducible(self):
        n=24;position=np.repeat([[48.,48.]],n,axis=0)
        sites=np.array([[20.,20.],[20.,80.],[80.,20.],[80.,80.],[48.,48.]])
        world=(self.collider,sites,np.ones((96,96,8)))
        rules=dict(RULES,pathfinding=False,vision_occlusion=False)
        pellets=np.repeat([[[40.,48.],[48.,40.],[56.,48.]]],n,axis=0)
        available=np.ones((n,3),bool);active=np.ones(n,bool)
        a=RandomPreyTargets([21],n);b=RandomPreyTargets([21],n)
        policy=OrganismPolicy(1,position,world,rules)
        first,_=a.choose(policy,position,pellets,available,0,active)
        same,_=b.choose(policy,position,pellets,available,0,active)
        np.testing.assert_array_equal(first,same)
        self.assertEqual(len(np.unique(first,axis=0)),3)
        np.testing.assert_array_equal(first,a.choose(policy,position,pellets,available,1,active)[0])
        # Exploration choices also differ, and hidden pellets cannot affect them.
        explore=RandomPreyTargets([21],n)
        empty=np.zeros_like(available)
        goals,_=explore.choose(policy,position,pellets,empty,0,active)
        self.assertGreater(len(np.unique(goals,axis=0)),1)


if __name__=='__main__':unittest.main()
