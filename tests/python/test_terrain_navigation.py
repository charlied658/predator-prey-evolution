"""Terrain routing must respect body clearance; sensing must respect occlusion."""
import unittest
import numpy as np
from src.terrain.terrain_navigation import TerrainNavigation
from src.terrain.grid_collision import GridCollider
from src.training.hybrid_pursuit import OrganismPolicy, RULES
from src.simulation.vision_pilot import GENES


class TerrainChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Finite wall, with room to pass its top or bottom.
        n=96
        data=dict(resolution=n,solid=[int(x==48 and 28<=y<=68) for y in range(n) for x in range(n)])
        collider=GridCollider(data,radius=2.4)
        sites=np.array([(x+.5,y+.5) for y in range(n) for x in range(n) if collider.clear((x+.5,y+.5))])
        cls.nav=TerrainNavigation(collider,sites)
        collider.navigation=cls.nav
        cls.world=(collider,sites,np.ones((96,96,8)))

    def test_wall_blocks_sight_but_not_open_space(self):
        self.assertFalse(self.nav.visible(np.array([40.,48.]),[56,48],36))
        self.assertTrue(self.nav.visible(np.array([40.,20.]),[56,20],36))
        self.assertFalse(self.nav.visible(np.array([10.,20.]),[56,20],36))

    def test_waypoints_route_around_wall_without_body_overlap(self):
        position=np.array([40.,48.]);goal=np.array([56.,48.]);length=0.
        for _ in range(40):
            waypoint=self.nav.waypoint(position,goal)
            self.assertTrue(self.nav.clear_segment(position,waypoint))
            length+=np.linalg.norm(waypoint-position)
            self.assertGreater(np.linalg.norm(waypoint-position),1e-6)
            position=waypoint
            if np.linalg.norm(position-goal)<1e-6:break
        np.testing.assert_allclose(position,goal)
        self.assertGreater(length,16)

    def test_occluded_prey_and_food_are_not_targeted(self):
        own=np.array([[40.,48.]])
        opponent=np.array([[56.,48.]])
        food=np.array([[[56.,50.]]])
        for role in (0,1):
            p=OrganismPolicy(role,own,self.world,RULES)
            p.command(np.zeros((1,GENES)),own,opponent,np.zeros((1,18)),food,np.ones((1,1),bool),0,np.ones(1,bool))
            self.assertFalse(p.seen[0])
            self.assertEqual(p.modes[0],'explore')

    def test_occlusion_switch_restores_circular_detection(self):
        p=OrganismPolicy(0,np.array([[40.,48.]]),self.world,dict(RULES,vision_occlusion=False))
        p.command(np.zeros((1,GENES)),np.array([[40.,48.]]),np.array([[56.,48.]]),np.zeros((1,18)),np.array([[[56.,50.]]]),np.ones((1,1),bool),0,np.ones(1,bool))
        self.assertTrue(p.seen[0])
        self.assertEqual(p.modes[0],'pursue')

    def test_fast_segments_match_reference_sweeps(self):
        rng=np.random.default_rng(211)
        for radius in (0.,2.4):
            reference=GridCollider(dict(resolution=96,solid=self.world[0].solid),radius=radius)
            for _ in range(2000):
                a=rng.uniform(3,93,2);b=rng.uniform(-2,98,2)
                if not reference.clear(a):continue
                self.assertEqual(self.nav.segment_clear(a,b,radius),reference.sweep(a,b-a)[1] is None,(a,b,radius))

    def test_fast_segments_on_irregular_terrain_and_tangencies(self):
        rng=np.random.default_rng(715)
        data=dict(resolution=24,solid=(rng.random(24**2)<.12).astype(int).tolist())
        collider=GridCollider(data,radius=.7)
        sites=np.array([(x+.5,y+.5) for y in range(24) for x in range(24) if collider.clear((x+.5,y+.5))])
        nav=TerrainNavigation(collider,sites)
        for radius in (0.,.7,2.4):
            reference=GridCollider(data,radius=radius)
            for _ in range(1000):
                a=rng.uniform(0,24,2);b=rng.uniform(-1,25,2)
                if reference.clear(a):
                    self.assertEqual(nav.segment_clear(a,b,radius),reference.sweep(a,b-a)[1] is None)
        # Wall parallel, face contact at the endpoint, and motion into the wall.
        for a,b in (([45.6,40],[45.6,50]),([40,48],[45.6,48]),([45.6,48],[46,48])):
            self.assertEqual(self.nav.clear_segment(a,b),self.world[0].sweep(a,np.array(b)-a)[1] is None)


if __name__=='__main__':unittest.main()
