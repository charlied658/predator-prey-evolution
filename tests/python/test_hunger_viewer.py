import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import tkinter as tk
import unittest
from src.simulation.hunger_tree import BASELINE,PREDATOR_BASELINE,DEFAULTS,action,episode
from src.simulation.vision_pilot import setup
from src.viewer.pursuit_viewer import PursuitViewer,load_run


class HungerViewerChecks(unittest.TestCase):
    def test_coevolved_pair_and_fixed_control_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            (path/'terrain.json').write_text(json.dumps(dict(resolution=96,solid=[0]*96**2)))
            (path/'design.json').write_text(json.dumps(dict(config=dict(DEFAULTS,seconds=2),budget=dict(terrain_map=None))))
            saved=dict(champions={'0':BASELINE},history={'0/0':BASELINE},predator_champions={'0':PREDATOR_BASELINE},predator_history={'0/0':action('stay')},fixed_champions={'0':action('stay')},fixed_history={'0/0':action('stay')},baseline_predator=PREDATOR_BASELINE)
            (path/'trees.json').write_text(json.dumps(saved))
            root=tk.Tk();root.withdraw()
            try:
                viewer=PursuitViewer(root,path,autoplay=False)
                self.assertIn('0',viewer.predator_box['values'])
                self.assertEqual(viewer.prey_condition.get(),'Co-evolution')
                self.assertIn('PREDATOR',viewer.tree_widget.get('1.0','end'))
                viewer.predator_generation.set('0');self.assertTrue(viewer.restart());viewer.single_step()
                self.assertEqual(viewer.frame['modes'][0],'stay')
                viewer.prey_condition.set('Fixed predator control');viewer.condition_changed()
                self.assertEqual(viewer.predator_generation.get(),'Fixed baseline');viewer.single_step()
                self.assertEqual(viewer.frame['modes'][1],'stay')
                viewer.close()
            finally:
                try:root.destroy()
                except tk.TclError:pass

    def test_desktop_load_step_play_restart_and_map_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);terrain=dict(resolution=96,solid=[0]*96**2)
            (path/'terrain.json').write_text(json.dumps(terrain))
            config=dict(DEFAULTS,seconds=2)
            (path/'design.json').write_text(json.dumps(dict(config=config,budget=dict(terrain_map=None))))
            (path/'trees.json').write_text(json.dumps(dict(champions={'0':BASELINE},history={'0/0':action('stay')})))
            self.assertEqual(load_run(path)[2]['controller_kind'],'hunger-tree')
            root=tk.Tk();root.withdraw()
            try:
                viewer=PursuitViewer(root,path,autoplay=False)
                self.assertEqual(viewer.frame['hunger'],config['initial_hunger'])
                self.assertIn('Scripted baseline',viewer.generation_box['values'])
                self.assertIn('Stationary',viewer.generation_box['values'])
                self.assertEqual(str(viewer.prey_count_widget.cget('state')),'disabled')
                self.assertIn('IF',viewer.tree_widget.get('1.0','end'))
                viewer.single_step();self.assertEqual(viewer.frame['seconds'],.1)
                viewer.toggle();self.assertTrue(viewer.playing)
                for _ in range(30):viewer.advance()
                self.assertFalse(viewer.playing);self.assertIsNone(viewer.generator)
                self.assertEqual(viewer.frame['death'],'timeout')
                result=episode(BASELINE,90000000,setup(terrain),config)
                self.assertAlmostEqual(viewer.frame['hunger'],result['final_hunger'])
                self.assertEqual(viewer.frame['food'],[result['food']])
                viewer.toggle();self.assertTrue(viewer.playing);self.assertEqual(viewer.frame['seconds'],0)
                viewer.single_step();self.assertEqual(viewer.frame['seconds'],.1)
                viewer.generation.set('Stationary');self.assertTrue(viewer.restart())
                self.assertEqual(viewer.tree_widget.get('1.0','end').strip(),'stay')
                viewer.maps['Other']=dict(resolution=96,solid=[int(x==48 and 25<=y<=70) for y in range(96) for x in range(96)])
                viewer.map_name.set('Other');self.assertTrue(viewer.restart())
                self.assertGreater(sum(viewer.world[0].solid),0)
                viewer.close()
            finally:
                try:root.destroy()
                except tk.TclError:pass

    @unittest.skipUnless(shutil.which('node'),'Optional Node runtime required to exercise browser replay')
    def test_html_play_restarts_after_end(self):
        template=(Path(__file__).resolve().parents[2]/'src/viewer/hunger_tree_replay.html').read_text(encoding='utf-8')
        frames=[dict(seconds=t,prey=[48,48],predator=[70,70],hunger=10,food=0,pellets=[],target=[48,48],action='stay',death=None) for t in (0,.1,.2)]
        payload=dict(examples={'Test':dict(tree='stay',frames=frames)},terrain=dict(solid=[0]*96**2),config=DEFAULTS)
        code=template.split('<script>',1)[1].split('</script>',1)[0].replace('/*PAYLOAD*/',json.dumps(payload))
        harness="""
const assert=require('node:assert/strict');let now=0;
const context=new Proxy({}, {get:()=>()=>{}});
const elements={};for(const id of ['canvas','example','play','restart','speed','vision','target','seek','time','tree','status','action'])elements[id]={value:'',checked:false,textContent:'',add(o){this.value=o.value}};
elements.canvas.getContext=()=>context;elements.speed.value='1';
global.document={getElementById:id=>elements[id]};global.Option=function(label,value){this.value=value};global.performance={now:()=>now};global.requestAnimationFrame=()=>{};
"""
        checks="""
assert.equal(playing,false);elements.play.onclick();assert.equal(playing,true);
now=100;loop(now);assert(t>0);now=1000;loop(now);assert.equal(playing,false);assert.equal(elements.play.textContent,'Play');
elements.play.onclick();assert.equal(t,0);assert.equal(playing,true);now=1100;loop(now);assert(t>0);
elements.play.onclick();const paused=t;now=1200;loop(now);assert.equal(t,paused);
elements.restart.onclick();assert.equal(t,0);assert.equal(playing,true);
"""
        with tempfile.TemporaryDirectory() as directory:
            script=Path(directory)/'replay-test.cjs';script.write_text(harness+code+checks,encoding='utf-8')
            result=subprocess.run([shutil.which('node'),str(script)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
