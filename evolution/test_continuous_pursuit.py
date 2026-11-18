import unittest
from unittest.mock import patch
import numpy as np
from continuous_pursuit import spawn_site,reset_individual,continuous_episodes,food_spawn_weights,spawn_food
from hybrid_pursuit import OrganismPolicy,RULES
from grid_collision import GridCollider
from vision_pilot import GENES,DT


class ContinuousChecks(unittest.TestCase):
    def setUp(self):
        self.world=(GridCollider(dict(resolution=96,solid=[0]*96**2)),np.array([[48.,48.]]),np.ones((96,96,8)))
        self.rules=dict(RULES,prey_count=2,respawn=True,memory_half_life=30.,respawn_delay=1.,pathfinding=False,vision_occlusion=False)

    def test_spawn_strict_exclusion_and_no_legal_site(self):
        rng=np.random.default_rng(3);sites=np.array([[10.,10.],[50.,50.],[80.,80.]])
        p=spawn_site(rng,sites,[[10.,10.],[80.,80.]],36,[],2.4)
        np.testing.assert_array_equal(p,[50,50])
        self.assertIsNone(spawn_site(rng,sites,sites,36,[],2.4))
        self.assertIsNone(spawn_site(rng,sites,[],0,sites,2.4))

    def test_food_soft_bias_has_no_vision_cutoff(self):
        sites=np.array([[6.,10.],[46.,10.]])
        prey=np.array([[0.,10.]])
        weights=food_spawn_weights(sites,prey,12)
        self.assertGreater(weights[0],0)
        self.assertGreater(weights[1],weights[0])
        rules=dict(self.rules,food_spawn_mode='soft',food_spawn_scale=12)
        rng=np.random.default_rng(172);chosen=[]
        for _ in range(600):chosen.append(spawn_food(rng,sites,prey,prey,2.4,[],rules)[0])
        fraction=np.mean(np.array(chosen)==46)
        self.assertGreater(fraction,.7);self.assertLess(fraction,.99)
        self.assertIsNone(spawn_food(rng,sites,prey,sites,2.4,[],rules))
        self.assertGreaterEqual(spawn_food(rng,sites,prey,prey,2.4,[],dict(rules,food_spawn_mode='strict'))[0],36)

    def test_memory_expires_and_respawn_resets_only_one_individual(self):
        p=OrganismPolicy(1,np.array([[20.,20.],[20.,20.]]),self.world,self.rules)
        original=p.visited[0].copy()
        p.mark(np.array([[70.,70.],[70.,70.]]),round(31/DT),np.ones(2,bool))
        self.assertFalse(np.any(p.visited[0]&original))
        other=p.last[1].copy()
        reset_individual(p,0,np.array([80.,20.]),400)
        np.testing.assert_array_equal(p.last[1],other)
        self.assertEqual(p.visited[0].sum(),1)
        self.assertEqual(p.last[0].max(),400)
        self.assertFalse(p.avoiding[0])

    def test_capture_respawns_and_pellet_is_awarded_once(self):
        # Captured prey subsequently stays still after the spawn.
        def commands(policy,genes,position,*args,**kwargs):
            if policy.role==0:return np.array([[18.,0.]])
            return np.where((position[:,0]<60)[:,None],np.array([[-18.,0.],[0.,0.]]),0.)
        placements=[np.array(p,float) for p in ([53,48],[20,20],[20,21],[80,80],[80,20])]
        with patch('continuous_pursuit.spawn_site',side_effect=placements),patch.object(OrganismPolicy,'command',commands):
            trace=[]
            result=continuous_episodes(np.zeros((1,GENES)),np.zeros((1,GENES)),[1],self.world,self.rules,2.5,1,8,trace,False,None)
        self.assertEqual(result[0,2],1)
        self.assertEqual(result[0,4],.5)
        self.assertGreater(result[0,3],.7)
        events=[e for f in trace for e in f['events']]
        self.assertEqual(sum(e['type']=='prey_respawn' for e in events),1)
        self.assertEqual(sum(e['type']=='food_respawn' for e in events),1)
        self.assertEqual(sum(e['type']=='capture' for e in events),1)
        self.assertTrue(trace[-1]['active'][0])

    def test_stationary_prey_cannot_collect_distant_food(self):
        placements=[np.array(p,float) for p in ([20,20],[80,80],[20,80])]
        with patch('continuous_pursuit.spawn_site',side_effect=placements),patch.object(OrganismPolicy,'command',lambda p,*a,**kw:np.zeros((p.n,2))):
            result=continuous_episodes(np.zeros((1,GENES)),np.zeros((1,GENES)),[1],self.world,self.rules,1,1,8,None,False,None)
        self.assertEqual(result[0,4],0)
        self.assertEqual(result[0,2],0)

    def test_two_prey_cannot_both_claim_one_pellet(self):
        placements=iter([np.array(p,float) for p in ([20,20],[25,20],[22.5,20])])
        with patch('continuous_pursuit.spawn_site',side_effect=lambda *a,**kw:next(placements,None)),patch.object(OrganismPolicy,'command',lambda p,*a,**kw:np.zeros((p.n,2))):
            trace=[]
            result=continuous_episodes(np.zeros((1,GENES)),np.zeros((1,GENES)),[1],self.world,self.rules,2,1,8,trace,False,None)
        self.assertEqual(result[0,4],.5)
        self.assertEqual(trace[-1]['food'],[1.,0.])

    def test_no_legal_respawn_keeps_prey_pending(self):
        placements=iter([np.array(p,float) for p in ([53,48],[20,20],[20,80])])
        def commands(policy,genes,position,*args,**kwargs):
            return np.array([[18.,0.]]) if policy.role==0 else np.array([[-18.,0.],[0.,0.]])
        with patch('continuous_pursuit.spawn_site',side_effect=lambda *a,**kw:next(placements,None)),patch.object(OrganismPolicy,'command',commands):
            trace=[]
            result=continuous_episodes(np.zeros((1,GENES)),np.zeros((1,GENES)),[1],self.world,self.rules,2.5,1,8,trace,False,None)
        self.assertEqual(result[0,2],1)
        self.assertFalse(trace[-1]['active'][0])
        self.assertEqual(trace[-1]['modes'][1],'respawn pending')


if __name__=='__main__':unittest.main()
