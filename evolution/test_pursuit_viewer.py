"""Verify stepping parity and that live viewer controls change the right state."""
import json
from pathlib import Path
import tempfile
import tkinter as tk
import unittest
import numpy as np
from continuous_pursuit import continuous_steps,continuous_episodes
from pursuit_viewer import PursuitViewer,load_terrain
from hybrid_pursuit import RULES,HYBRID_INSTRUCTIONS
from vision_pilot import GENES,setup


class ViewerChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.path=Path(cls.temp.name)
        cls.terrain=dict(resolution=96,solid=[0]*96**2)
        cls.rules=dict(RULES,prey_count=2,respawn=True,memory_half_life=30,prey_collisions=True,random_prey_targets=True)
        cls.genomes=np.zeros((2,GENES))
        (cls.path/'terrain.json').write_text(json.dumps(cls.terrain))
        (cls.path/'champions.json').write_text(json.dumps({'0':cls.genomes.tolist()}))
        (cls.path/'history.json').write_text(json.dumps({'0/0':cls.genomes.tolist()}))
        (cls.path/'instructions.json').write_text(json.dumps(dict(pursuit=cls.rules,foraging_and_recovery=HYBRID_INSTRUCTIONS)))

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_generator_matches_batch_and_emits_initial_state(self):
        world=setup(self.terrain);args=(self.genomes[0][None],self.genomes[1][None],[44],world,self.rules,1)
        trace=[];expected=continuous_episodes(*args,12,8,trace,False,None)
        streamed=continuous_steps(*args,emit=True);frames=[]
        while True:
            try:frames.append(next(streamed))
            except StopIteration as finished:actual=finished.value;break
        np.testing.assert_array_equal(expected,actual)
        self.assertEqual(frames,trace)
        self.assertEqual(len(frames),11)
        self.assertEqual(frames[0]['seconds'],0)
        self.assertEqual(frames[-1]['seconds'],1)

    def test_controls_restart_map_and_display_without_changing_physics(self):
        root=tk.Tk();root.withdraw()
        try:
            viewer=PursuitViewer(root,self.path,autoplay=False);root.update_idletasks()
            initial=json.dumps(viewer.frame);viewer.single_step()
            self.assertEqual(viewer.frame['seconds'],.1)
            before=json.dumps(viewer.frame)
            viewer.flags['Vision'].set(0);viewer.overlay_changed();viewer.global_flags['Grid lines'].set(True);viewer.draw()
            self.assertFalse(viewer.canvas.find_withtag('vision'))
            self.assertEqual(len(viewer.canvas.find_withtag('grid')),18)
            self.assertEqual(json.dumps(viewer.frame),before)
            self.assertTrue(viewer.restart());self.assertEqual(json.dumps(viewer.frame),initial)
            viewer.new_encounter();self.assertNotEqual(json.dumps(viewer.frame),initial)
            wall=dict(resolution=96,solid=[int(x==48 and 25<=y<=70) for y in range(96) for x in range(96)])
            viewer.maps['Test wall']=wall;viewer.map_name.set('Test wall');self.assertTrue(viewer.restart())
            self.assertGreater(sum(viewer.world[0].solid),0)
            self.assertEqual(viewer.frame['seconds'],0)
            viewer.memory_for.set('All prey');viewer.focus_changed()
            viewer.flags['Vision'].set(1);viewer.overlay_changed()
            self.assertTrue(viewer.display_settings[1]['Vision'])
            self.assertTrue(viewer.display_settings[2]['Vision'])
            self.assertFalse(viewer.display_settings[0]['Vision'])
            viewer.memory_for.set('Prey 1');viewer.focus_changed()
            viewer.flags['Vision'].set(0);viewer.overlay_changed()
            self.assertFalse(viewer.canvas.find_withtag('vision-1'))
            self.assertTrue(viewer.canvas.find_withtag('vision-2'))
            viewer.memory_for.set('All prey');viewer.focus_changed()
            self.assertEqual(viewer.flags['Vision'].get(),-1)
            viewer.openness.set(.8);self.assertTrue(viewer.generate_map())
            self.assertEqual(viewer.terrain['generator'],'multiscale-value-noise-v1')
            first=viewer.terrain['solid'].copy();self.assertTrue(viewer.new_map())
            self.assertNotEqual(first,viewer.terrain['solid'])
            self.assertEqual(viewer.active_rules['food_spawn_mode'],'soft')
            viewer.food_mode.set('Strict outside vision');self.assertTrue(viewer.restart())
            self.assertEqual(viewer.active_rules['food_spawn_mode'],'strict')
            viewer.close()
        finally:
            try:root.destroy()
            except tk.TclError:pass

    def test_invalid_map_is_rejected(self):
        path=self.path/'invalid.json';path.write_text(json.dumps(dict(resolution=12,solid=[0]*144)))
        with self.assertRaisesRegex(ValueError,'96 x 96'):load_terrain(path)


if __name__=='__main__':unittest.main()
