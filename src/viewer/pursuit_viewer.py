"""Live desktop viewer for saved pursuit controllers. No training or server."""

from src.paths import ROOT
import argparse
import secrets

from collections import deque, OrderedDict

import json

from pathlib import Path

import time

import tkinter as tk

from tkinter import ttk,filedialog,messagebox

import numpy as np

from PIL import Image,ImageTk

from src.simulation.continuous_pursuit import continuous_steps

from src.training.hybrid_pursuit import RULES,HYBRID_INSTRUCTIONS

from src.simulation.vision_pilot import GENES,DT,setup

from src.terrain.terrain_navigation import navigation

from src.terrain.procedural_terrain import generate_terrain





COLORS=['#df755e','#239b9d','#3972bf','#aa5cbb','#8e9237','#af7736','#278b67']





def load_terrain(path):

    terrain=json.loads(Path(path).read_text())

    if terrain.get('resolution')!=96 or len(terrain.get('solid',[]))!=96**2:

        raise ValueError('Choose a 96 x 96 exported terrain JSON file')

    if any(v not in (0,1,False,True) for v in terrain['solid']):raise ValueError('Terrain cells must be 0 or 1')

    if not any(v==0 for v in terrain['solid']):raise ValueError('Map has no open ground')

    return terrain





def load_run(path):

    path=Path(path).resolve()

    champions=json.loads((path/'champions.json').read_text())

    valid={str(k):np.asarray(v,float) for k,v in champions.items()}

    if not valid or any(v.shape!=(2,GENES) or not np.isfinite(v).all() for v in valid.values()):

        raise ValueError('Choose a hybrid pursuit run with saved predator/prey controllers')

    saved=json.loads((path/'instructions.json').read_text())

    if saved.get('foraging_and_recovery')!=HYBRID_INSTRUCTIONS:

        raise ValueError('Saved recovery rules differ from this simulation version')

    rules=dict(RULES,**saved['pursuit'])

    rules.update(respawn=True,prey_count=rules.get('prey_count',3),memory_half_life=rules.get('memory_half_life',30),respawn_delay=rules.get('respawn_delay',1))

    # Earlier runs may be evaluated in this continuous sandbox; show the actual

    # enabled rules rather than silently claiming it reproduces their old task.

    # Legacy runs predate this flag. The interactive sandbox should use solid
    # prey bodies for them too; direct historical evaluations still retain the
    # engine's old default. Explicit collision-off comparison runs stay off.
    rules.setdefault('prey_collisions',True);rules.setdefault('random_prey_targets',False)

    history=json.loads((path/'history.json').read_text()) if (path/'history.json').exists() else {}

    return path,valid,rules,history,load_terrain(path/'terrain.json')





def latest_run():

    candidates=[p for p in (ROOT/'results').glob('*') if (p/'champions.json').exists() and (p/'instructions.json').exists()]

    for p in sorted(candidates,key=lambda p:(p/'champions.json').stat().st_mtime,reverse=True):

        try:load_run(p);return p

        except (ValueError,KeyError,TypeError,OSError):continue

    raise ValueError('No compatible saved pursuit run found; supply its directory as the argument')





class PursuitViewer:

    def __init__(self,root,run,autoplay=True):

        self.root=root;root.title('Predator / prey — live simulation viewer');root.geometry('1220x880');root.minsize(1000,740)

        self.playing=autoplay;self.frame=None;self.generator=None;self.trail=deque(maxlen=101)

        self.worlds=OrderedDict();self.outlines=OrderedDict();self.photo=None;self.closed=False;self.after_id=None

        self.speed=tk.StringVar(value='1');self.seed=tk.StringVar(value='90000000');self.duration=tk.StringVar(value='120')

        self.prey_count=tk.StringVar(value='3');self.controller=tk.StringVar();self.generation=tk.StringVar(value='Final');self.map_name=tk.StringVar(value='Training map')

        self.memory_for=tk.StringVar(value='Prey 1');self.status=tk.StringVar(value='Loading controllers…');self.stats=tk.StringVar();self.rule_text=tk.StringVar()

        self.flags={name:tk.IntVar(value=int(name in ('Vision','Route waypoints','Visit memory','Labels'))) for name in ('Vision','Route waypoints','Final targets','Visit memory','Trails','Labels')}

        self.global_flags={name:tk.BooleanVar(value=default) for name,default in [('Grid lines',False),('Food',True)]}


        self.map_seed=tk.StringVar(value='42');self.map_kind=tk.StringVar(value='Outcrops')

        self.openness=tk.DoubleVar(value=.7);self.roughness=tk.DoubleVar(value=.35);self.feature_size=tk.DoubleVar(value=16.)

        self.map_details=tk.StringVar();self.food_mode=tk.StringVar(value='Soft distance bias');self.food_scale=tk.StringVar(value='12')

        self._controls();self.load(run)

        root.protocol('WM_DELETE_WINDOW',self.close)

        root.bind('<space>',self.toggle_key);root.bind('<Right>',self.step_key)

        self.after_id=root.after(25,self.tick)



    def _controls(self):

        style=ttk.Style();style.theme_use('clam')

        outer=ttk.Frame(self.root,padding=14);outer.pack(fill='both',expand=True)

        sidebar=ttk.Frame(outer,width=320);sidebar.pack(side='left',fill='y',padx=(0,16));sidebar.pack_propagate(False)

        ttk.Label(sidebar,text='Pursuit sandbox',font=('Segoe UI',18,'bold')).pack(anchor='w')

        ttk.Label(sidebar,text='Saved controllers · live evaluation',foreground='#47665e').pack(anchor='w',pady=(0,8))

        tabs=ttk.Notebook(sidebar);tabs.pack(fill='both',expand=True)

        run=ttk.Frame(tabs,padding=10);maps=ttk.Frame(tabs,padding=10);display=ttk.Frame(tabs,padding=10)

        tabs.add(run,text='Run');tabs.add(maps,text='Map & food');tabs.add(display,text='Overlays')

        def combo(parent,label,var,values,command):

            ttk.Label(parent,text=label).pack(anchor='w',pady=(8,2))

            widget=ttk.Combobox(parent,textvariable=var,values=values,state='readonly')

            widget.pack(fill='x');widget.bind('<<ComboboxSelected>>',lambda e:command());return widget

        self.run_label=ttk.Label(run,wraplength=285);self.run_label.pack(anchor='w')

        ttk.Button(run,text='Load another run…',command=self.choose_run).pack(fill='x',pady=6)

        self.controller_box=combo(run,'Controller seed',self.controller,[],self.controller_changed)

        self.generation_box=combo(run,'Saved generation',self.generation,['Final'],self.restart)

        form=ttk.Frame(run);form.pack(fill='x',pady=12)

        for row,(label,var,low,high) in enumerate([('Encounter seed',self.seed,0,2**31-1),('Duration (s)',self.duration,1,3600),('Prey count',self.prey_count,1,12)]):

            ttk.Label(form,text=label).grid(row=row,column=0,sticky='w',pady=4)

            ttk.Spinbox(form,textvariable=var,from_=low,to=high,width=13).grid(row=row,column=1,sticky='e',padx=8)

        buttons=ttk.Frame(run);buttons.pack(fill='x',pady=6)

        self.play_button=ttk.Button(buttons,text='Pause' if self.playing else 'Play',command=self.toggle);self.play_button.pack(side='left',expand=True,fill='x')

        ttk.Button(buttons,text='Step',command=self.single_step).pack(side='left',expand=True,fill='x',padx=(5,0))

        ttk.Button(run,text='Restart same encounter',command=self.restart).pack(fill='x',pady=4)

        ttk.Button(run,text='New encounter (next seed)',command=self.new_encounter).pack(fill='x',pady=4)

        combo(run,'Playback speed',self.speed,['0.25','0.5','1','2','4'],lambda:None)

        ttk.Label(run,text='Prey count and duration apply on restart.\nSpace: play/pause · Right arrow: one step.',wraplength=285).pack(anchor='w',pady=14)

        self.map_box=combo(maps,'Terrain',self.map_name,[],self.restart)

        row=ttk.Frame(maps);row.pack(fill='x',pady=5)

        ttk.Button(row,text='Load JSON…',command=self.choose_map).pack(side='left',expand=True,fill='x')

        ttk.Button(row,text='Save map…',command=self.save_map).pack(side='left',expand=True,fill='x',padx=(4,0))

        ttk.Separator(maps).pack(fill='x',pady=10)

        ttk.Label(maps,text='Generate terrain',font=('Segoe UI',11,'bold')).pack(anchor='w')

        combo(maps,'Type',self.map_kind,['Outcrops','Ridges','Clearings'],lambda:None)

        ttk.Label(maps,text='Map seed').pack(anchor='w',pady=(8,2))

        ttk.Spinbox(maps,textvariable=self.map_seed,from_=0,to=2**31-1).pack(fill='x')

        for label,var,low,high in [('Openness',self.openness,.35,.95),('Roughness',self.roughness,0,1),('Feature size',self.feature_size,6,32)]:

            value=tk.StringVar();value.set(f'{label}: {var.get():.2f}')

            ttk.Label(maps,textvariable=value).pack(anchor='w',pady=(8,0))

            ttk.Scale(maps,variable=var,from_=low,to=high,command=lambda v,var=var,label=label,value=value:value.set(f'{label}: {float(v):.2f}')).pack(fill='x')

        row=ttk.Frame(maps);row.pack(fill='x',pady=10)

        ttk.Button(row,text='Generate',command=self.generate_map).pack(side='left',expand=True,fill='x')

        ttk.Button(row,text='Random map seed',command=self.new_map).pack(side='left',expand=True,fill='x',padx=(4,0))

        ttk.Label(maps,textvariable=self.map_details,wraplength=285).pack(anchor='w')

        ttk.Separator(maps).pack(fill='x',pady=10)

        combo(maps,'Food placement',self.food_mode,['Soft distance bias','Strict outside vision'],self.restart)

        ttk.Label(maps,text='Food distance scale (cells)').pack(anchor='w',pady=(7,2))

        ttk.Spinbox(maps,textvariable=self.food_scale,from_=1,to=96).pack(fill='x')

        ttk.Button(maps,text='Apply food settings / restart',command=self.restart).pack(fill='x',pady=7)

        ttk.Label(maps,text='Soft bias: nearby food is less likely, not forbidden. Larger scale discourages a wider area.',wraplength=285).pack(anchor='w')

        self.memory_box=combo(display,'Selected organism',self.memory_for,['Predator','Prey 1'],self.focus_changed)

        ttk.Label(display,text='Show these indicators only for the selected organism. Switching organisms moves the indicators and keeps your checkbox choices.',wraplength=285).pack(anchor='w',pady=9)

        for name,var in self.flags.items():

            ttk.Checkbutton(display,text=name,variable=var,command=self.overlay_changed).pack(anchor='w',pady=4)

        ttk.Button(display,text='Hide all organism overlays',command=self.clear_overlays).pack(fill='x',pady=10)

        ttk.Separator(display).pack(fill='x',pady=8)

        ttk.Label(display,text='Global display',font=('Segoe UI',11,'bold')).pack(anchor='w')

        for name,var in self.global_flags.items():ttk.Checkbutton(display,text=name,variable=var,command=self.draw).pack(anchor='w',pady=4)

        ttk.Label(display,text='Click an organism to focus it. Display settings never change physics or sensing.',wraplength=285).pack(anchor='w',pady=14)

        ttk.Label(sidebar,textvariable=self.rule_text,wraplength=315,foreground='#47665e').pack(anchor='w',pady=(10,0))

        right=ttk.Frame(outer);right.pack(side='left',fill='both',expand=True)

        ttk.Label(right,textvariable=self.stats,font=('Segoe UI',11),justify='left').pack(anchor='w',pady=(0,6))

        self.canvas=tk.Canvas(right,background='#172321',highlightthickness=0);self.canvas.pack(fill='both',expand=True)

        self.canvas.bind('<Configure>',lambda e:self.draw());self.canvas.bind('<Button-1>',self.select_organism)

        ttk.Label(right,textvariable=self.status,wraplength=800).pack(anchor='w',pady=(6,0))



    def load(self,run):

        path,champions,rules,history,terrain=load_run(run)

        self.run=path;self.champions=champions;self.rules=rules;self.history=history

        self.run_label.config(text=path.name)

        seeds=sorted(champions,key=int);self.controller_box['values']=seeds;self.controller.set(seeds[0])

        self.prey_count.set(str(rules['prey_count']))

        self.food_mode.set('Strict outside vision' if rules.get('food_spawn_mode')=='strict' else 'Soft distance bias')

        self.food_scale.set(str(rules.get('food_spawn_scale',12.)))

        self.maps={'Training map':terrain,'Empty arena':dict(resolution=96,solid=[0]*96**2,preset='Empty arena')}

        for file in sorted((ROOT/'results/landscape-previews').glob('*.json')):

            try:

                candidate=load_terrain(file)

                name=str(candidate.get('preset',file.stem))+' · '+file.stem

                self.maps[name]=candidate

            except (ValueError,OSError):continue

        self.map_box['values']=list(self.maps);self.map_name.set('Training map')

        self.controller_changed()



    def controller_changed(self):

        values=['Final']+[str(g) for g in sorted(int(key.split('/')[1]) for key in self.history if key.split('/')[0]==self.controller.get())]

        self.generation_box['values']=values

        if self.generation.get() not in values:self.generation.set('Final')

        self.restart()

    def generate_map(self):
        try:
            terrain=generate_terrain(int(self.map_seed.get()),self.openness.get(),self.roughness.get(),self.feature_size.get(),self.map_kind.get())
            world=setup(terrain)
            if len(world[1])<32:raise ValueError('Too little body-clear ground; increase openness or feature size')
            navigation(world);self.worlds[bytes(terrain['solid'])]=world
            if len(self.worlds)>4:self.worlds.popitem(last=False)
            self.maps['Generated terrain']=terrain;self.map_box['values']=list(self.maps);self.map_name.set('Generated terrain')
            self.map_details.set(f'Open ground: {terrain["open_fraction"]:.1%}\nBody-accessible cells: {len(world[1])}/9216')
            return self.restart()
        except Exception as error:
            self.status.set('Cannot generate map: '+str(error));return False

    def new_map(self):
        self.map_seed.set(str(secrets.randbelow(2**31)))
        return self.generate_map()

    def save_map(self):
        path=filedialog.asksaveasfilename(title='Save selected terrain',defaultextension='.json',filetypes=[('Terrain JSON','*.json')],initialfile='terrain.json')
        if path:
            try:Path(path).write_text(json.dumps(self.maps[self.map_name.get()]),encoding='utf-8');self.status.set('Saved map: '+path)
            except OSError as error:messagebox.showerror('Cannot save map',str(error),parent=self.root)



    def choose_run(self):

        directory=filedialog.askdirectory(title='Choose a saved hybrid pursuit run',initialdir=ROOT/'results')

        if directory:

            try:self.load(directory)

            except Exception as error:messagebox.showerror('Cannot load run',str(error),parent=self.root)



    def choose_map(self):

        path=filedialog.askopenfilename(title='Choose a 96 x 96 terrain mask',initialdir=ROOT/'results',filetypes=[('Terrain JSON','*.json')])

        if not path:return

        try:

            terrain=load_terrain(path);setup(terrain) # Validate accessible body space before replacing the map.

            name='Custom · '+Path(path).name;self.maps[name]=terrain

            self.map_box['values']=list(self.maps);self.map_name.set(name);self.restart()

        except Exception as error:messagebox.showerror('Cannot load map',str(error),parent=self.root)



    def restart(self):

        try:

            seed=int(self.seed.get());seconds=float(self.duration.get());count=int(self.prey_count.get())

            if not 0<=seed<2**63 or not 1<=count<=12 or not 0<seconds<=3600 or abs(round(seconds/DT)*DT-seconds)>1e-8:

                raise ValueError('Use a nonnegative integer seed, 1–12 prey, and a duration up to 3600 seconds in 0.1-second steps')

            terrain=self.maps[self.map_name.get()];key=bytes(terrain['solid'])

            if key not in self.worlds:

                world=setup(terrain);navigation(world);self.worlds[key]=world

                if len(self.worlds)>4:self.worlds.popitem(last=False)

            self.worlds.move_to_end(key);world=self.worlds[key]

            seed_key=self.controller.get();generation=self.generation.get()

            genomes=self.champions[seed_key] if generation=='Final' else np.asarray(self.history[f'{seed_key}/{generation}'],float)

            scale=float(self.food_scale.get())

            if not 0<scale<=96:raise ValueError('Food distance scale must be in (0,96]')

            rules=dict(self.rules,prey_count=count,food_spawn_mode='soft' if self.food_mode.get()=='Soft distance bias' else 'strict',food_spawn_scale=scale)

            generator=continuous_steps(genomes[0][None],genomes[1][None],[seed],world,rules,seconds,emit=True)

            first=next(generator)

        except Exception as error:

            self.playing=False;self.play_button.config(text='Play')

            self.status.set('Cannot restart: '+str(error));return False

        if self.generator is not None:self.generator.close()

        self.generator=generator;self.world=world;self.terrain=terrain;self.active_rules=rules;self.solid=np.asarray(terrain['solid'],bool).reshape(96,96)

        self.base=Image.fromarray(np.where(self.solid[:,:,None],[108,119,116],[232,228,206]).astype('uint8')).convert('RGBA')

        self.frame=first;self.trail.clear();self.trail.append(first);self.outlines.clear()

        self.ensure_display(count)

        self.rule_text.set(f'Vision: predator {rules["predator_vision"]:g}, prey {rules["prey_vision"]:g}\nOcclusion: {"on" if rules.get("vision_occlusion") else "off"} · collisions: {"on" if rules.get("prey_collisions") else "off"}\nRandom targets: {"on" if rules.get("random_prey_targets") else "off"} · food: {rules['food_spawn_mode']}')

        self.status.set('Live continuous evaluation · map changes restart the encounter · display toggles do not change physics')

        self.draw();return True



    def new_encounter(self):

        try:self.seed.set(str(int(self.seed.get())+1));self.restart()

        except ValueError:self.status.set('Enter an integer encounter seed')



    def toggle(self):

        if self.generator is None and not self.playing:

            if not self.restart():return

        self.playing=not self.playing;self.play_button.config(text='Pause' if self.playing else 'Play')



    def toggle_key(self,event):

        if not isinstance(self.root.focus_get(),(ttk.Entry,ttk.Spinbox,ttk.Combobox,ttk.Button,ttk.Checkbutton)):self.toggle();return 'break'



    def step_key(self,event):

        if not isinstance(self.root.focus_get(),(ttk.Entry,ttk.Spinbox,ttk.Combobox)):self.single_step();return 'break'



    def advance(self):

        if self.generator is None:return

        try:self.frame=next(self.generator);self.trail.append(self.frame)

        except StopIteration:

            self.generator=None;self.playing=False;self.play_button.config(text='Play');self.status.set('Encounter finished. Restart it or choose a new encounter. No training was performed.')

        except Exception as error:

            self.generator=None;self.playing=False;self.play_button.config(text='Play');self.status.set('Simulation stopped: '+str(error))

        self.draw()



    def single_step(self):

        self.playing=False;self.play_button.config(text='Play');self.advance()



    def tick(self):

        start=time.perf_counter()

        if self.playing:self.advance()

        if not self.closed:

            elapsed=(time.perf_counter()-start)*1000

            delay=max(1,round(DT*1000/float(self.speed.get())-elapsed)) if self.playing else 50

            self.after_id=self.root.after(delay,self.tick)



    def scope_indices(self):
        return [0 if self.memory_for.get()=='Predator' else int(self.memory_for.get().split()[-1])]

    def ensure_display(self,count):
        self.memory_box['values']=['Predator']+[f'Prey {j+1}' for j in range(count)]
        if self.memory_for.get() not in self.memory_box['values']:
            self.memory_for.set('Prey 1' if count else 'Predator')

    def focus_changed(self):
        self.draw()

    def overlay_changed(self):
        self.draw()

    def clear_overlays(self):
        for var in self.flags.values():var.set(0)
        self.draw()

    def focus_index(self):return self.scope_indices()[0]



    def select_organism(self,event):

        self.canvas.focus_set()

        if self.frame is None:return

        size,ox,oy=self.geometry()

        point=np.array([(event.x-ox)*96/size,(event.y-oy)*96/size])

        positions=np.asarray(self.frame['positions']);dist=np.linalg.norm(positions-point,axis=1)

        for j,active in enumerate(self.frame['active'],1):

            if not active:dist[j]=np.inf

        index=int(dist.argmin())

        if dist[index]<5:self.memory_for.set('Predator' if index==0 else f'Prey {index}');self.focus_changed()



    def geometry(self):

        w=max(300,self.canvas.winfo_width());h=max(300,self.canvas.winfo_height());size=max(260,min(w,h)-20)

        return size,(w-size)/2,(h-size)/2



    def draw(self):
        if self.frame is None or not hasattr(self,'base'):return
        f=self.frame;c=self.canvas;c.delete('all');size,ox,oy=self.geometry();scale=size/96;selected=set(self.scope_indices())
        image=self.base;confidence=np.zeros(64)
        for j in selected:
            if self.flags['Visit memory'].get():
                confidence=np.maximum(confidence,f['predator_memory'] if j==0 else f['memory'][j-1])
        if confidence.any():
            overlay=np.zeros((96,96,4),dtype='uint8')
            for cell,value in enumerate(confidence):
                x,y=cell%8,cell//8;overlay[y*12:(y+1)*12,x*12:(x+1)*12]=[20,170,80,round(100*value)]
            overlay[self.solid,3]=0;image=Image.alpha_composite(image,Image.fromarray(overlay))
        self.photo=ImageTk.PhotoImage(image.resize((round(size),round(size)),Image.Resampling.NEAREST))
        c.create_image(ox,oy,image=self.photo,anchor='nw',tags='terrain')
        def coords(point):return ox+point[0]*scale,oy+point[1]*scale
        if self.global_flags['Grid lines'].get():
            for j in range(9):
                q=j*size/8;c.create_line(ox+q,oy,ox+q,oy+size,fill='#9da995',tags='grid');c.create_line(ox,oy+q,ox+size,oy+q,fill='#9da995',tags='grid')
        if self.global_flags['Food'].get():
            for point in f['remaining']:
                x,y=coords(point);r=max(3,.8*scale);c.create_oval(x-r,y-r,x+r,y+r,fill='#e3b72f',outline='#66551a',tags='food')
        for focus in selected:
            if not self.flags['Trails'].get():continue
            segment=[]
            for frame in self.trail:
                active=focus==0 or frame['active'][focus-1]
                respawn=any(e['type']=='prey_respawn' and e['prey']==focus-1 for e in frame['events']) if focus else False
                if not active or respawn:
                    if len(segment)>=4:c.create_line(*segment,fill=COLORS[focus%len(COLORS)],width=2,tags=('trail',f'trail-{focus}'))
                    segment=[]
                if active:segment.extend(coords(frame['positions'][focus]))
            if len(segment)>=4:c.create_line(*segment,fill=COLORS[focus%len(COLORS)],width=2,tags=('trail',f'trail-{focus}'))
        for j,point in enumerate(f['positions']):
            if j and not f['active'][j-1]:continue
            flags={name:j in selected and bool(var.get()) for name,var in self.flags.items()};color=COLORS[j%len(COLORS)];x,y=coords(point)
            if flags['Vision']:
                radius=self.active_rules['predator_vision' if j==0 else 'prey_vision']
                if self.active_rules.get('vision_occlusion',False):
                    key=tuple(point)+(radius,)
                    if key not in self.outlines:
                        self.outlines[key]=navigation(self.world).vision_outline(point,radius)
                        if len(self.outlines)>256:self.outlines.popitem(last=False)
                    points=[coords(p) for p in self.outlines[key]]
                else:points=[coords(np.clip(np.asarray(point)+radius*np.array([np.cos(a),np.sin(a)]),0,96)) for a in np.arange(64)*np.pi/32]
                points.append(points[0]);c.create_line(*[v for p in points for v in p],fill=color,tags=('vision',f'vision-{j}'))
            if flags['Route waypoints']:
                gx,gy=coords(f['goals'][j]);c.create_line(x,y,gx,gy,fill=color,dash=(4,3),tags=('waypoint',f'waypoint-{j}'));c.create_oval(gx-3,gy-3,gx+3,gy+3,outline=color,tags=('waypoint',f'waypoint-{j}'))
            if flags['Final targets']:
                gx,gy=coords(f['targets'][j]);c.create_line(x,y,gx,gy,fill=color,stipple='gray50',tags=('target',f'target-{j}'));c.create_line(gx-5,gy,gx+5,gy,fill=color,tags='target');c.create_line(gx,gy-5,gx,gy+5,fill=color,tags='target')
            # Tk centres strokes on the oval boundary. Inset the border so its
            # outer edge, including selection highlighting, matches the body.
            width=3 if j in selected else 1
            r=max(0.,self.world[0].radius*scale-width/2)
            c.create_oval(x-r,y-r,x+r,y+r,fill=color,outline='white' if j in selected else '#172321',width=width,tags='organism')
            if flags['Labels']:c.create_text(x,y-r-9,text='Predator' if j==0 else f'Prey {j}',fill='#172321',font=('Segoe UI',10,'bold'),tags=('label',f'label-{j}'))
        modes=' | '.join(('P' if j==0 else str(j))+': '+mode for j,mode in enumerate(f['modes']))
        self.stats.set(f'Time {f["seconds"]:.1f}s   |   Captures {f["captures"]}   |   Food {sum(f["food"]):g}   |   Active prey {sum(f["active"])}/{len(f["active"])}\n{modes}')

    def close(self):

        self.closed=True

        if self.after_id is not None:self.root.after_cancel(self.after_id)

        if self.generator is not None:self.generator.close()

        self.root.destroy()





def main():

    parser=argparse.ArgumentParser(description=__doc__)

    parser.add_argument('run',type=Path,nargs='?',help='Saved hybrid pursuit run; defaults to the latest compatible run')

    parser.add_argument('--paused',action='store_true')

    args=parser.parse_args()

    try:run=args.run or latest_run();load_run(run)

    except (OSError,ValueError,KeyError,TypeError) as error:parser.error(str(error))

    root=tk.Tk();PursuitViewer(root,run,autoplay=not args.paused);root.mainloop()





if __name__=='__main__':main()

