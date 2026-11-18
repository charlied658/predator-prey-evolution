"""Check vision cutoffs and masking of all opponent-specific information."""
import unittest
from unittest.mock import patch
import vision_pilot as pilot
import numpy as np
from vision_pilot import observe, food_observation, collect_food, training_ids, food_layout, starts

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

if __name__=='__main__':unittest.main()
