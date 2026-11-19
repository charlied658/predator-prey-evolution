"""Check vision cutoffs and masking of all opponent-specific information."""
import unittest
from unittest.mock import patch
import src.simulation.vision_pilot as pilot
import numpy as np
from src.simulation.vision_pilot import observe, food_observation, collect_food, training_ids, food_layout, starts

class VisionChecks(unittest.TestCase):
    def test_independent_ranges_and_boundary(self):
        rays=np.zeros((96,96,8));p=np.array([[[10.,10.],[34.,10.]]]);v=np.zeros_like(p)
        obs,seen=observe(p,v,(24,48),rays)
        self.assertTrue(seen.all())
        p[0,1,0]+=0.01
        obs,seen=observe(p,v,(24,48),rays)
        self.assertEqual(seen.tolist(),[[False,True]])
        self.assertTrue(np.all(obs[0][0,:5]==0))
        self.assertGreater(obs[1][0,4],0)
    def test_hidden_opponent_location_and_velocity_do_not_leak(self):
        rays=np.zeros((96,96,8));p=np.array([[[10.,10.],[70.,70.]]]);v=np.zeros_like(p)
        first,_=observe(p,v,(24,24),rays)
        p[0,1]=[80,80];v[0,1]=[10,-10]
        second,_=observe(p,v,(24,24),rays)
        np.testing.assert_array_equal(first[0],second[0])

class FoodChecks(unittest.TestCase):
    def test_food_sensing_masks_hidden_and_consumed_pellets(self):
        pos=np.array([[0.,0.]])
        pellets=np.array([[[10.,0.],[30.,0.]]])
        available=np.array([[True,True]])
        self.assertEqual(food_observation(pos,pellets,available,24)[0,2],1)
        available[0,0]=False
        np.testing.assert_array_equal(food_observation(pos,pellets,available,24),np.zeros((1,3)))
        self.assertEqual(food_observation(pos,pellets,available,48)[0,2],1)

    def test_swept_collection_once_and_only_while_active(self):
        old=np.array([[0.,0.]]);new=np.array([[20.,0.]])
        pellets=np.array([[[10.,0.],[10.,10.]]]);available=np.array([[True,True]])
        self.assertEqual(collect_food(old,new,pellets,available,np.array([False]))[0],0)
        self.assertEqual(collect_food(old,new,pellets,available,np.array([True]))[0],1)
        self.assertEqual(collect_food(old,new,pellets,available,np.array([True]))[0],0)

class EncounterChecks(unittest.TestCase):
    def test_schedule_is_reproducible_and_separate_from_evaluation(self):
        np.testing.assert_array_equal(training_ids(1,1),training_ids(1,1,'fixed'))
        np.testing.assert_array_equal(training_ids(1,2),training_ids(1,2))
        self.assertFalse(np.array_equal(training_ids(1,2),training_ids(1,3)))
        np.testing.assert_array_equal(training_ids(1,2,'fixed'),training_ids(1,3,'fixed'))
        self.assertFalse(set(training_ids(1,2)) & set(range(5100,5112)))

    def test_schedule_changes_both_positions_and_pellets(self):
        sites=np.array([(x+.5,y+.5) for x in range(5,90) for y in range(5,90)])
        ids1=training_ids(0,2);ids2=training_ids(0,3)
        pos1=starts(ids1,sites);pos2=starts(ids2,sites)
        self.assertFalse(np.array_equal(pos1,pos2))
        food1=food_layout(ids1,sites,pos1);food2=food_layout(ids2,sites,pos2)
        self.assertFalse(np.array_equal(food1,food2))
        np.testing.assert_array_equal(food1,food_layout(ids1,sites,pos1))

class ForagingModeChecks(unittest.TestCase):
    def test_no_capture_or_opponent_information_but_food_still_collected(self):
        body=pilot.GridCollider(dict(resolution=96,solid=[0]*96**2),radius=2.4)
        world=(body,np.array([[50.,50.]]),np.zeros((96,96,8)))
        pellets=np.array([[[50.,50.]]+[[80.,80.]]*11])
        positions=np.array([[[50.,50.],[50.,50.]]])
        def stationary(genomes,obs):
            np.testing.assert_array_equal(obs[:,:5],np.zeros((len(obs),5)))
            return np.zeros((len(obs),2))
        genome=np.zeros((1,pilot.GENES))
        with patch.object(pilot,'starts',return_value=positions.copy()), patch.object(pilot,'food_layout',return_value=pellets), patch.object(pilot,'network',side_effect=stationary):
            result=pilot.episodes(genome,genome,[0],(24,24),world,capture=False)
        self.assertEqual(result[0,2],0)
        self.assertEqual(result[0,3],pilot.STEPS*pilot.DT)
        self.assertEqual(result[0,6],1)

class FoodEncodingChecks(unittest.TestCase):
    def test_unit_direction_and_hidden_food_mask(self):
        p=np.array([[10.,10.]]);pellets=np.array([[[13.,14.]]]);active=np.array([[True]])
        np.testing.assert_allclose(food_observation(p,pellets,active,24,'unit'),[[.6,.8,1.]])
        np.testing.assert_allclose(food_observation(p,pellets,active,24,'original'),[[3/96,4/96,1.]])
        np.testing.assert_array_equal(food_observation(p,pellets,active,2,'unit'),np.zeros((1,3)))
        np.testing.assert_array_equal(food_observation(p,p[:,None],active,24,'unit'),[[0,0,1]])

class CaptureChecks(unittest.TestCase):
    def test_capture_is_restored_with_unit_food_inputs(self):
        collider=pilot.GridCollider(dict(resolution=96,solid=[0]*96**2),radius=2.4)
        world=(collider,np.array([[50.,50.]]),np.zeros((96,96,8)))
        positions=np.array([[[50.,50.],[50.,50.]]])
        pellets=np.array([[[80.,80.]]*12]);genome=np.zeros((1,pilot.GENES))
        with patch.object(pilot,'starts',return_value=positions),patch.object(pilot,'food_layout',return_value=pellets):
            result=pilot.episodes(genome,genome,[0],(24,24),world,capture=True,food_encoding='unit')
        self.assertEqual(result[0,2],1)
        self.assertEqual(result[0,3],pilot.DT)

class TrainingCountChecks(unittest.TestCase):
    def test_eight_encounters_include_the_two_encounter_schedule(self):
        for generation in (1,2,50):
            small=training_ids(1,generation,count=2)
            large=training_ids(1,generation,count=8)
            np.testing.assert_array_equal(small,large[:2])
            self.assertEqual(len(set(large)),8)
        self.assertFalse(set(training_ids(0,2,count=8)) & set(training_ids(1,2,count=8)))

class ConfidenceIntervalChecks(unittest.TestCase):
    def test_constant_seed_trajectories_have_zero_width(self):
        mean,low,high=pilot.seed_mean_interval([[1,2],[1,2],[1,2]])
        np.testing.assert_array_equal(mean,[1,2])
        np.testing.assert_array_equal(low,mean)
        np.testing.assert_array_equal(high,mean)

    def test_bootstrap_preserves_whole_seed_trajectories(self):
        mean,low,high=pilot.seed_mean_interval([[1,2],[3,6],[5,10]])
        np.testing.assert_allclose(mean,[3,6])
        self.assertAlmostEqual(low[1],2*low[0])
        self.assertAlmostEqual(high[1],2*high[0])
        self.assertLess(low[0],high[0])
        with self.assertRaises(ValueError):pilot.seed_mean_interval([[float('nan')]])

class PredatorMaskChecks(unittest.TestCase):
    def test_mask_removes_only_prey_opponent_inputs_and_preserves_capture(self):
        collider=pilot.GridCollider(dict(resolution=96,solid=[0]*96**2),radius=2.4)
        world=(collider,np.array([[50.,50.]]),np.zeros((96,96,8)))
        positions=np.array([[[50.,50.],[50.,50.]]]);pellets=np.array([[[60.,50.]]*12])
        calls=[]
        def stationary(genomes,obs):
            calls.append(obs.copy());return np.zeros((len(obs),2))
        genome=np.zeros((1,pilot.GENES))
        with patch.object(pilot,'starts',return_value=positions),patch.object(pilot,'food_layout',return_value=pellets),patch.object(pilot,'network',side_effect=stationary):
            result=pilot.episodes(genome,genome,[0],(24,24),world,mask_prey_opponent=True)
        self.assertEqual(calls[0][0,4],1)
        np.testing.assert_array_equal(calls[1][0,:5],np.zeros(5))
        np.testing.assert_allclose(calls[1][0,-3:],[10/96,0,1])
        self.assertEqual(result[0,2],1)

class SinglePelletChecks(unittest.TestCase):
    def test_random_pellet_is_visible_and_not_already_collectable(self):
        from src.simulation.single_pellet import arrangements
        positions,pellets=arrangements(range(128))
        distance=np.linalg.norm(pellets[:,0]-positions,axis=1)
        self.assertTrue(np.all((distance>=6)&(distance<=20)))
        self.assertTrue(np.all((pellets>2.4)&(pellets<93.6)))
        np.testing.assert_array_equal(positions,arrangements(range(128))[0])

    def test_direct_steering_succeeds_and_stationary_network_does_not(self):
        from src.simulation.single_pellet import evaluate
        world=pilot.setup(dict(resolution=96,solid=[0]*96**2))
        g=np.zeros((16,pilot.GENES));ids=list(range(16))
        reference=evaluate(g,ids,world,policy='direct')
        stationary=evaluate(g,ids,world)
        self.assertTrue(reference[:,0].all())
        self.assertFalse(stationary[:,0].any())
        self.assertTrue(np.all(reference[:,1]<2))

class MultiplePelletChecks(unittest.TestCase):
    def test_larger_sets_preserve_starts_and_existing_pellets(self):
        from src.simulation.single_pellet import arrangements
        p1,f1=arrangements(range(16),1);p4,f4=arrangements(range(16),4);p12,f12=arrangements(range(16),12)
        np.testing.assert_array_equal(p1,p4);np.testing.assert_array_equal(p1,p12)
        np.testing.assert_array_equal(f1,f4[:,:1]);np.testing.assert_array_equal(f4,f12[:,:4])
        self.assertTrue(np.all(np.linalg.norm(f12-p12[:,None],axis=-1)<=24))
        for i in range(12):
            for j in range(i):self.assertTrue(np.all(np.linalg.norm(f12[:,i]-f12[:,j],axis=1)>=6))

    def test_food_collection_continues_after_first_pellet(self):
        from src.simulation.single_pellet import evaluate
        world=pilot.setup(dict(resolution=96,solid=[0]*96**2));g=np.zeros((1,pilot.GENES))
        position=np.array([[48.,48.]]);pellets=np.array([[[54.,48.],[60.,48.],[66.,48.],[68.,48.]]])
        with patch('src.simulation.single_pellet.arrangements',return_value=(position,pellets)):
            result=evaluate(g,[0],world,policy='direct',pellet_count=4)
        self.assertEqual(result[0,2],4)
        self.assertEqual(result[0,0],1)

class ScatteredFoodChecks(unittest.TestCase):
    def test_scattered_food_is_clear_reachable_reproducible_and_not_all_local(self):
        import json
        from pathlib import Path
        from src.simulation.single_pellet import arrangements
        # Use the actual landscape being tested; this check validates placement
        # against finite body size rather than merely empty map cells.
        from src.terrain.procedural_terrain import generate_terrain
        terrain=generate_terrain(seed=42,openness=.8)
        collider,sites,_=pilot.setup(terrain)
        positions,pellets=arrangements(range(32),12,sites)
        p2,f2=arrangements(range(32),12,sites)
        np.testing.assert_array_equal(positions,p2);np.testing.assert_array_equal(pellets,f2)
        for point in np.concatenate((positions,pellets.reshape(-1,2))):self.assertTrue(collider.clear(point))
        self.assertTrue(np.any(np.linalg.norm(pellets-positions[:,None],axis=-1)>24))
        self.assertTrue(np.all(pellets.reshape(-1,2).max(axis=0)-pellets.reshape(-1,2).min(axis=0)>80))
        self.assertTrue(np.all(np.linalg.norm(pellets-positions[:,None],axis=-1)>pilot.FOOD_RADIUS+2))

class ExplorationGridChecks(unittest.TestCase):
    def test_starting_tile_only_and_revisits_not_double_counted(self):
        from src.simulation.single_pellet import evaluate
        world=pilot.setup(dict(resolution=96,solid=[0]*96**2));g=np.zeros((1,pilot.GENES));food=np.array([[[80.,80.]]])
        def layout(*args):return np.array([[11.5,18.]]),food.copy()
        with patch('src.simulation.single_pellet.arrangements',side_effect=layout):
            still=evaluate(g,[0],world)
        self.assertEqual(still[0,3],1)
        calls=[0]
        def alternate(genomes,obs):
            self.assertEqual(obs.shape,(1,pilot.INPUTS))
            calls[0]+=1
            return np.array([[1. if calls[0]%2 else -1.,0.]])
        with patch('src.simulation.single_pellet.arrangements',side_effect=layout),patch('src.simulation.single_pellet.network',side_effect=alternate):
            roaming=evaluate(g,[0],world)
        self.assertEqual(roaming[0,3],2)
        self.assertEqual(roaming[0,2],0)
        self.assertAlmostEqual(.1*(roaming[0,3]-1),.1)

class RecurrentControllerChecks(unittest.TestCase):
    def test_hidden_state_carries_a_cue_and_reset_removes_it(self):
        from src.simulation.single_pellet import recurrent_network,REC_HIDDEN,REC_GENES
        g=np.zeros((1,REC_GENES));cut=(pilot.INPUTS+1)*REC_HIDDEN
        g[0,15*REC_HIDDEN]=1
        g[0,cut]=1
        g[0,cut+REC_HIDDEN**2]=1
        cue=np.zeros((1,pilot.INPUTS));cue[0,15]=1
        zero=np.zeros((1,REC_HIDDEN));_,state=recurrent_network(g,cue,zero)
        retained,_=recurrent_network(g,np.zeros_like(cue),state)
        reset,_=recurrent_network(g,np.zeros_like(cue),zero)
        self.assertGreater(retained[0,0],.1)
        np.testing.assert_array_equal(reset,np.zeros((1,2)))
        np.testing.assert_array_equal(zero,np.zeros_like(zero))
        self.assertEqual(REC_GENES,164)

    def test_recurrent_encounters_do_not_share_state(self):
        from src.simulation.single_pellet import evaluate,REC_GENES
        world=pilot.setup(dict(resolution=96,solid=[0]*96**2))
        g=np.random.default_rng(12).normal(0,.45,(2,REC_GENES));ids=[123,456]
        batch=evaluate(g,ids,world,controller='recurrent')
        split=np.vstack([evaluate(g[i:i+1],[ids[i]],world,controller='recurrent') for i in range(2)])
        np.testing.assert_allclose(batch,split)
        np.testing.assert_allclose(batch,evaluate(g,ids,world,controller='recurrent'))

class ExplicitMapChecks(unittest.TestCase):
    def test_blanking_map_preserves_position_and_local_inputs(self):
        from src.simulation.single_pellet import map_observation,map_gene_count
        obs=np.arange(36,dtype=float).reshape(2,18);pos=np.array([[24.,48.],[72.,24.]])
        visited=np.zeros((2,64),bool);visited[0,3]=True;visited[1,12]=True
        visible=map_observation(obs,pos,visited);blank=map_observation(obs,pos,visited,True)
        self.assertEqual(visible.shape,(2,84));self.assertEqual(map_gene_count(8),698)
        np.testing.assert_array_equal(visible[:,:20],blank[:,:20])
        np.testing.assert_array_equal(visible[:,:18],obs)
        np.testing.assert_array_equal(visible[:,18:20],pos/48-1)
        np.testing.assert_array_equal(blank[:,20:],np.zeros((2,64)))
        self.assertEqual(visible[0,23],1)
        self.assertEqual(visible[1,32],1)

    def test_map_state_resets_per_encounter_and_does_not_cross_batch_rows(self):
        from src.simulation.single_pellet import evaluate,map_gene_count
        world=pilot.setup(dict(resolution=96,solid=[0]*96**2))
        g=np.random.default_rng(32).normal(0,.1,(2,map_gene_count()))
        ids=[123,456];batch=evaluate(g,ids,world,controller='visited-map')
        split=np.vstack([evaluate(g[i:i+1],[ids[i]],world,controller='visited-map') for i in range(2)])
        np.testing.assert_allclose(batch,split)
        np.testing.assert_allclose(batch,evaluate(g,ids,world,controller='visited-map'))
        np.testing.assert_allclose(evaluate(g,ids,world,controller='visited-map',blank_map=True),evaluate(g,ids,world,controller='position-only'))

class ExperimentReuseChecks(unittest.TestCase):
    def test_reuse_signature_changes_with_reward_or_world(self):
        from types import SimpleNamespace
        from src.simulation.single_pellet import run_compatibility,map_gene_count
        args=SimpleNamespace(generations=1,seeds=2,food_layout='scattered',grid_cells=8,exploration_reward=0.)
        world=pilot.setup(dict(resolution=96,solid=[0]*96**2))
        one=run_compatibility(args,12,world,'position-only',map_gene_count(),49)
        same=run_compatibility(args,12,world,'position-only',map_gene_count(),49)
        self.assertEqual(one['fingerprint'],same['fingerprint'])
        args.exploration_reward=.1
        other=run_compatibility(args,12,world,'position-only',map_gene_count(),49)
        self.assertNotEqual(one['fingerprint'],other['fingerprint'])
        args.exploration_reward=0.;world[0].radius=2.5
        other=run_compatibility(args,12,world,'position-only',map_gene_count(),49)
        self.assertNotEqual(one['fingerprint'],other['fingerprint'])

    def test_incomplete_control_is_rejected(self):
        import tempfile,csv
        from pathlib import Path
        from types import SimpleNamespace
        from src.simulation.single_pellet import load_run_rows
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            with (path/'measurements.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['seed','generation','food','visited','success','capped_time','fraction','fitness']);writer.writeheader()
                writer.writerow(dict(seed=0,generation=0,food=1,visited=1,success=0,capped_time=10,fraction=.1,fitness=1))
            with self.assertRaisesRegex(ValueError,'incomplete'):
                load_run_rows(path,SimpleNamespace(seeds=2,generations=1))

class HybridInstructionChecks(unittest.TestCase):
    def test_food_priority_persistent_exploration_and_no_unseen_food_leak(self):
        from src.simulation.single_pellet import hybrid_target
        pos=np.array([[48.,48.]]);visited=np.zeros((1,64),bool);visited[0,36]=True
        last=np.full((1,64),-1);last[0,36]=0;target=np.array([1])
        food=np.array([[[60.,48.],[80.,80.]]]);available=np.ones((1,2),bool)
        goals,forage=hybrid_target(pos,food,available,visited,last,target)
        np.testing.assert_array_equal(goals,[[60,48]]);self.assertTrue(forage[0]);self.assertEqual(target[0],-1)
        available[0,0]=False
        goals,forage=hybrid_target(pos,food,available,visited,last,target)
        old=goals.copy();chosen=target[0];self.assertFalse(forage[0])
        # Entering its tile does not prematurely discard a persistent goal.
        visited[0,chosen]=True
        retained,_=hybrid_target(pos,food,available,visited,last,target)
        np.testing.assert_array_equal(retained,old)
        # Unseen pellets cannot affect a newly selected exploration target.
        visited[0,chosen]=False;target[:]=-1
        first,_=hybrid_target(pos,food,available,visited,last,target)
        target[:]=-1;food[0,1]=[10,80]
        second,_=hybrid_target(pos,food,available,visited,last,target)
        np.testing.assert_array_equal(first,second)

    def test_heading_and_speed_constraints(self):
        from src.simulation.single_pellet import hybrid_command
        raw=np.array([[-1,-1],[0,0],[1,1]])
        command=hybrid_command(raw,np.zeros((3,2)),np.tile([10.,0.],(3,1)))
        speed=np.linalg.norm(command,axis=1)
        self.assertTrue(np.all(speed>=.2-1e-12));self.assertTrue(np.all(speed<=1+1e-12))
        self.assertTrue(np.all(command[:,0]/speed>=.5-1e-12))

    def test_direct_hybrid_searches_when_food_is_outside_vision(self):
        from src.simulation.single_pellet import evaluate
        world=pilot.setup(dict(resolution=96,solid=[0]*96**2));g=np.zeros((1,pilot.GENES))
        def layout(*args):return np.array([[48.,48.]]),np.array([[[85.,85.]]])
        with patch('src.simulation.single_pellet.arrangements',side_effect=layout):
            guided=evaluate(g,[0],world,policy='direct',controller='hybrid')
            reactive=evaluate(g,[0],world,policy='direct')
        self.assertGreater(guided[0,3],1)
        self.assertEqual(reactive[0,3],1)

class HybridTerrainChecks(unittest.TestCase):
    def test_exploration_goals_are_clear_and_blocked_tiles_are_ignored(self):
        import json
        from pathlib import Path
        from src.simulation.single_pellet import hybrid_exploration_sites,hybrid_target
        from src.terrain.procedural_terrain import generate_terrain
        world=pilot.setup(generate_terrain(seed=42,openness=.8))
        centres,eligible=hybrid_exploration_sites(world)
        self.assertTrue(eligible.any())
        for point in centres[eligible]:self.assertTrue(world[0].clear(point))
        selected=np.flatnonzero(eligible)[-1]
        visited=np.ones((1,64),bool);last=np.zeros((1,64),int);last[0,~eligible]=-1000;last[0,selected]=-10
        goal,food=hybrid_target(world[1][0:1],np.array([[[50.,50.]]]),np.zeros((1,1),bool),visited,last,np.array([-1]),8,(centres,eligible))
        np.testing.assert_array_equal(goal[0],centres[selected]);self.assertFalse(food[0])

class LongerEncounterChecks(unittest.TestCase):
    def test_duration_override_preserves_default_and_records_final_frame(self):
        from src.simulation.single_pellet import evaluate
        world=pilot.setup(dict(resolution=96,solid=[0]*96**2));g=np.zeros((1,pilot.GENES));trace=[]
        short=evaluate(g,[123],world,duration_seconds=2,trace=trace)
        normal=evaluate(g,[123],world)
        self.assertEqual(short[0,1],2)
        self.assertEqual(normal[0,1],10)
        self.assertEqual(len(trace),21)
        self.assertEqual(trace[-1]['contacts'],0)
        with self.assertRaises(ValueError):evaluate(g,[123],world,duration_seconds=0)

class StuckRecoveryChecks(unittest.TestCase):
    def test_escape_waypoint_has_clear_collision_sweep(self):
        from src.simulation.single_pellet import hybrid_escape_waypoint
        solid=[int(x==54 and y<70) for y in range(96) for x in range(96)]
        collider=pilot.GridCollider(dict(resolution=96,solid=solid),radius=2.4)
        pos=np.array([51.5,48.]);goal=hybrid_escape_waypoint(collider,pos,np.array([60.,48.]))
        self.assertTrue(collider.clear(goal));fraction,normal=collider.sweep(pos,goal-pos)
        self.assertIsNone(normal);self.assertEqual(fraction,1)
        self.assertLess(goal[0],pos[0])

    def test_stalled_controller_recovers_and_old_rule_can_be_reproduced(self):
        from src.simulation.single_pellet import evaluate
        solid=[int(x==54 and y<70) for y in range(96) for x in range(96)]
        world=pilot.setup(dict(resolution=96,solid=solid));g=np.zeros((1,pilot.GENES))
        def layout(*args):return np.array([[50.,48.]]),np.array([[[60.,48.]]])
        before=[];after=[]
        with patch('src.simulation.single_pellet.arrangements',side_effect=layout):
            normal=evaluate(g,[0],world,controller='hybrid',trace=before,duration_seconds=8,recovery=False)
            recovered=evaluate(g,[0],world,controller='hybrid',trace=after,duration_seconds=8,recovery=True)
        self.assertEqual(before[-1]['recoveries'],0)
        self.assertGreater(after[-1]['recoveries'],0)
        self.assertTrue(any(fr.get('mode')=='recover' for fr in after))
        self.assertGreater(recovered[0,3],normal[0,3])
        self.assertGreaterEqual(next(i for i,fr in enumerate(after) if fr.get('mode')=='recover')*pilot.DT,2)
        for fr in after:self.assertTrue(world[0].clear(fr['position']))

class ProgressSamplingChecks(unittest.TestCase):
    def test_progress_reports_final_values_without_affecting_results(self):
        from src.simulation.single_pellet import evaluate
        world=pilot.setup(dict(resolution=96,solid=[0]*96**2));g=np.zeros((1,pilot.GENES));progress=[]
        sampled=evaluate(g,[123],world,policy='direct',duration_seconds=2,progress=progress)
        original=evaluate(g,[123],world,policy='direct',duration_seconds=2)
        np.testing.assert_array_equal(sampled,original)
        self.assertEqual(progress[0]['seconds'],0)
        self.assertEqual(progress[-1]['seconds'],sampled[0,1])
        self.assertEqual(progress[-1]['food'][0],sampled[0,2])
        self.assertEqual(progress[-1]['visited'][0],sampled[0,3])
        self.assertEqual(len(progress),2)

if __name__=='__main__':unittest.main()
