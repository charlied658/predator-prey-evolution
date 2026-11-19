"""Verify stepping parity and that live viewer controls change the right state."""
import json
from pathlib import Path
import tempfile
import tkinter as tk
import unittest
import numpy as np
from src.simulation.continuous_pursuit import continuous_steps,continuous_episodes
from src.viewer.pursuit_viewer import PursuitViewer,load_terrain,load_run
from src.training.hybrid_pursuit import RULES,HYBRID_INSTRUCTIONS
from src.simulation.vision_pilot import GENES,setup


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
            # Include half the centred stroke on either side: painted body
            # diameter must match physics even with the selected white border.
            scale=viewer.geometry()[0]/96
            for item in viewer.canvas.find_withtag('organism'):
                x0,y0,x1,y1=viewer.canvas.coords(item)
                width=float(viewer.canvas.itemcget(item,'width'))
                self.assertAlmostEqual(x1-x0+width,2*viewer.world[0].radius*scale)
                self.assertAlmostEqual(y1-y0+width,2*viewer.world[0].radius*scale)
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
            self.assertEqual(tuple(viewer.memory_box['values']),('Predator','Prey 1','Prey 2'))
            for var in viewer.flags.values():var.set(1)
            viewer.memory_for.set('Prey 1');viewer.focus_changed()
            state=json.dumps(viewer.frame)
            for selection,index in [('Prey 1',1),('Prey 2',2),('Predator',0),('Prey 1',1)]:
                viewer.memory_for.set(selection);viewer.focus_changed()
                for tag in ('vision','waypoint','target','label'):
                    self.assertTrue(viewer.canvas.find_withtag(f'{tag}-{index}'))
                    for other in range(3):
                        if other!=index:self.assertFalse(viewer.canvas.find_withtag(f'{tag}-{other}'))
                self.assertEqual(json.dumps(viewer.frame),state)
                self.assertTrue(all(var.get()==1 for var in viewer.flags.values()))
            viewer.flags['Vision'].set(0);viewer.overlay_changed()
            self.assertFalse(viewer.canvas.find_withtag('vision'))
            viewer.memory_for.set('Prey 2');viewer.focus_changed()
            self.assertFalse(viewer.canvas.find_withtag('vision'))
            viewer.clear_overlays()
            for tag in ('vision','waypoint','target','label','trail'):
                self.assertFalse(viewer.canvas.find_withtag(tag))
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

    def test_legacy_controller_run_enables_viewer_body_collisions(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            for name in ('terrain.json','champions.json'):
                (path/name).write_bytes((self.path/name).read_bytes())
            legacy=dict(self.rules);legacy.pop('prey_collisions')
            instructions=path/'instructions.json'
            instructions.write_text(json.dumps(dict(pursuit=legacy,foraging_and_recovery=HYBRID_INSTRUCTIONS)))
            original=instructions.read_bytes()
            self.assertTrue(load_run(path)[2]['prey_collisions'])
            self.assertEqual(instructions.read_bytes(),original)
            legacy['prey_collisions']=False
            instructions.write_text(json.dumps(dict(pursuit=legacy,foraging_and_recovery=HYBRID_INSTRUCTIONS)))
            self.assertFalse(load_run(path)[2]['prey_collisions'])


if __name__=='__main__':unittest.main()
