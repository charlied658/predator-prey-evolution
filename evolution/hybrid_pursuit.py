"""Explicit hybrid predator/prey priorities with evolving movement controllers."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
from single_pellet import (HYBRID_INSTRUCTIONS,arrangements,hybrid_target,
    hybrid_exploration_sites,hybrid_escape_waypoint,hybrid_command)
from vision_pilot import GENES,DT,network,observe,collect_food,setup,seed_mean_interval

from terrain_navigation import navigation, TerrainNavigation

RULES=json.loads(Path(__file__).with_name('hybrid_pursuit_instructions.json').read_text())


def interpolate_path(path,times):
    ts=np.array([x[0] for x in path]);points=np.array([x[1] for x in path])
    return np.c_[np.interp(times,ts,points[:,0]),np.interp(times,ts,points[:,1])]


def capture_fraction(predator_path,prey_path,radius):
    """Earliest contact along actual collision-resolved piecewise motion."""
    times=np.unique([t for t,_ in predator_path]+[t for t,_ in prey_path])
    a=interpolate_path(predator_path,times);b=interpolate_path(prey_path,times)
    rel=a-b
    for i in range(len(times)-1):
        start=rel[i];delta=rel[i+1]-start;c=float(start@start-radius**2)
        if c<=1e-10:return float(times[i])
        length=float(delta@delta);dot=float(start@delta)
        discriminant=dot*dot-length*c
        if length>1e-18 and dot<0 and discriminant>=0:
            u=(-dot-np.sqrt(discriminant))/length
            if 0<=u<=1:return float(times[i]+u*(times[i+1]-times[i]))
    if np.linalg.norm(rel[-1])<=radius+1e-10:return float(times[-1])
    return None


def flee_waypoint(collider,position,predator):
    """Choose a local clear direction, preferring nondecreasing separation."""
    away=position-predator;distance=max(np.linalg.norm(away),1e-12);away/=distance
    candidates=[]
    for angle in np.arange(16)*np.pi/8:
        direction=np.array([np.cos(angle),np.sin(angle)]);fraction,_=collider.sweep(position,direction*12)
        clearance=max(0,12*fraction-1e-4)
        if clearance>.25:
            goal=position+direction*clearance;radial=float(direction@away)
            score=float(np.linalg.norm(goal-predator)-distance+.2*clearance)
            candidates.append((radial>=-1e-8,score,goal))
    if not candidates:return position.copy()
    safe=[item for item in candidates if item[0]]
    return max(safe or candidates,key=lambda x:x[1])[2]


def outward_command(command,position,predator,minimum):
    """Avoidance may not intentionally steer toward a currently seen threat."""
    away=position-predator;away/=np.maximum(np.linalg.norm(away,axis=1,keepdims=True),1e-12)
    speed=np.maximum(np.linalg.norm(command,axis=1),minimum)
    angle=np.arctan2(away[:,0]*command[:,1]-away[:,1]*command[:,0],np.sum(away*command,axis=1))
    angle=np.clip(angle,-np.deg2rad(85),np.deg2rad(85));c,s=np.cos(angle),np.sin(angle)
    return np.c_[away[:,0]*c-away[:,1]*s,away[:,0]*s+away[:,1]*c]*speed[:,None]


class OrganismPolicy:
    def __init__(self,role,position,world,rules,grid=8):
        self.nav=navigation(world) if rules.get('pathfinding',False) or rules.get('vision_occlusion',False) else None
        self.role=role;self.world=world;self.rules=rules;self.grid=grid;self.n=len(position)
        self.target_selector=None
        self.sites=hybrid_exploration_sites(world,grid)
        self.visited=np.zeros((self.n,grid**2),bool);self.last=np.full((self.n,grid**2),-1,int);self.explore=np.full(self.n,-1,int)
        self.window=round(HYBRID_INSTRUCTIONS['stuck_seconds']/DT)
        self.history=np.repeat(position[:,None],self.window+1,axis=1)
        self.until=np.zeros(self.n,int);self.next_check=np.full(self.n,self.window,int);self.escape=position.copy();self.recovery_avoids=np.zeros(self.n,bool)
        self.food_cooldown=None;self.tile_cooldown=np.zeros((self.n,grid**2),int);self.recoveries=np.zeros(self.n,int);self.avoiding=np.zeros(self.n,bool)
        self.goals=position.copy();self.modes=np.full(self.n,'explore',dtype='<U16');self.seen=np.zeros(self.n,bool)
        self.mark(position,0,np.ones(self.n,bool))

    def mark(self,position,tick,active):
        half_life=self.rules.get('memory_half_life',0)
        if half_life>0:self.visited &= (tick-self.last)*DT<half_life
        cells=np.clip((position*self.grid/96).astype(int),0,self.grid-1);ids=cells[:,1]*self.grid+cells[:,0];rows=np.flatnonzero(active)
        self.visited[rows,ids[rows]]=True;self.last[rows,ids[rows]]=tick
        self.history[:,tick%(self.window+1)]=position

    def command(self,genomes,position,opponent,obs,pellets,available,tick,active,scripted=False):
        collider=self.world[0];rules=self.rules;distance=np.linalg.norm(opponent-position,axis=1)
        vision=rules['predator_vision'] if self.role==0 else rules['prey_vision'];seen=distance<=vision
        if rules.get('vision_occlusion',False):
            for row in np.flatnonzero(seen):seen[row]=self.nav.visible(position[row],opponent[row],vision)
        self.seen=seen
        if self.food_cooldown is None:self.food_cooldown=np.zeros_like(available,dtype=int)
        planning_food=available&(self.food_cooldown<=tick) if self.role==1 else np.zeros_like(available)
        if rules.get('vision_occlusion',False):
            planning_food=planning_food.copy()
            for row,index in np.argwhere(planning_food & (np.linalg.norm(pellets-position[:,None],axis=-1)<=vision)):
                planning_food[row,index]=self.nav.visible(position[row],pellets[row,index],vision)
        goals,food=hybrid_target(position,pellets,planning_food,self.visited,self.last,self.explore,self.grid,self.sites,self.tile_cooldown>tick,food_vision=vision)
        if self.target_selector is not None:
            goals,food=self.target_selector.choose(self,position,pellets,planning_food,tick,active)
        modes=np.where(food,'forage','explore').astype('<U16')
        if self.role==0:
            pursue=seen;goals[pursue]=opponent[pursue];modes[pursue]='pursue';self.explore[pursue]=-1
        else:
            enter=rules['predator_vision']+rules['avoidance_margin'];release=enter+rules['avoidance_hysteresis']
            self.avoiding=seen & ((distance<=enter)|(self.avoiding&(distance<release)))
            # An urgent threat interrupts ordinary recovery immediately.
            cancel=self.avoiding & ~self.recovery_avoids & (self.until>tick);self.until[cancel]=tick
            for row in np.flatnonzero(active&self.avoiding):goals[row]=flee_waypoint(collider,position[row],opponent[row])
            modes[self.avoiding]='avoid';self.explore[self.avoiding]=-1
        reached=np.linalg.norm(self.escape-position,axis=1)<=HYBRID_INSTRUCTIONS['goal_reached_distance']
        self.until[reached&(self.until>tick)]=tick
        confined=np.linalg.norm(self.history-position[:,None],axis=-1).max(axis=1)<HYBRID_INSTRUCTIONS['stuck_radius']
        stuck=active & (tick>=self.window) & confined & (tick>=self.next_check) & (tick>=self.until)
        for row in np.flatnonzero(stuck):
            cooldown=tick+round(HYBRID_INSTRUCTIONS['target_cooldown_seconds']/DT)
            if modes[row]=='forage':
                index=np.linalg.norm(pellets[row]-goals[row],axis=1).argmin();self.food_cooldown[row,index]=cooldown
            elif modes[row]=='explore' and self.explore[row]>=0:self.tile_cooldown[row,self.explore[row]]=cooldown
            self.explore[row]=-1;avoid=self.role==1 and self.avoiding[row];self.recovery_avoids[row]=avoid
            self.escape[row]=flee_waypoint(collider,position[row],opponent[row]) if avoid else hybrid_escape_waypoint(collider,position[row],goals[row])
            self.until[row]=tick+round(HYBRID_INSTRUCTIONS['recovery_seconds']/DT);self.next_check[row]=self.until[row]+self.window;self.recoveries[row]+=1
        recovering=active&(tick<self.until)
        if self.role==1:
            # Update the escape goal if a moving predator makes it unsafe.
            unsafe=recovering&self.avoiding&(np.sum((self.escape-position)*(position-opponent),axis=1)<0)
            for row in np.flatnonzero(unsafe):self.escape[row]=flee_waypoint(collider,position[row],opponent[row])
        goals[recovering]=self.escape[recovering];modes[recovering]='recover'
        if self.role==1:modes[recovering&self.avoiding]='recover-avoid'
        self.targets=goals.copy()
        if rules.get('pathfinding',False):
            for row in np.flatnonzero(active):goals[row]=self.nav.waypoint(position[row],goals[row])
        inputs=obs.copy();delta=goals-position;d=np.linalg.norm(delta,axis=1)
        inputs[:,:5]=0
        inputs[:,0]=np.array([{'explore':0,'forage':.25,'pursue':.5,'avoid':.75,'recover':1,'recover-avoid':1}[mode] for mode in modes])
        inputs[:,1]=d/96;inputs[:,2]=seen;inputs[:,3]=np.where(seen,distance/96,0)
        inputs[:,4]=self.avoiding if self.role==1 else seen
        inputs[:,-3:-1]=delta/np.maximum(d[:,None],1e-12);inputs[:,-1]=d>1e-12
        command=inputs[:,-3:-1].copy() if scripted else hybrid_command(network(genomes,inputs),position,goals)
        if recovering.any():
            direction=delta[recovering]/np.maximum(d[recovering,None],1e-12)
            command[recovering]=direction*HYBRID_INSTRUCTIONS['recovery_throttle']
        if self.role==1 and self.avoiding.any():
            mask=self.avoiding;command[mask]=outward_command(command[mask],position[mask],opponent[mask],rules['avoidance_minimum_throttle'])
        if rules.get('pathfinding',False):
            for row in np.flatnonzero(active):
                if not self.nav.clear_segment(position[row],position[row]+command[row]*rules['speed']*DT):
                    direction=delta[row]/max(d[row],1e-12)
                    if self.role!=1 or not self.avoiding[row] or direction@(position[row]-opponent[row])>=0:
                        command[row]=direction*np.linalg.norm(command[row])
        command[~active]=0;self.goals=goals;self.modes=modes
        return command*rules['speed']


def make_arrangements(ids,world,count):
    prey,pellets=arrangements(ids,count,world[1]);predator=[]
    for seed,position in zip(ids,prey):
        rng=np.random.default_rng(int(seed)+70000077);distance=np.linalg.norm(world[1]-position,axis=1)
        candidates=world[1][(distance>=18)&(distance<=42)]
        if not len(candidates):raise ValueError('No valid predator spawn at distance 18..42')
        predator.append(candidates[rng.integers(len(candidates))])
    return np.stack((np.array(predator),prey),axis=1),pellets


def episodes(predators,prey_genomes,ids,world,rules,seconds=20,pellet_count=12,grid=8,trace=None,scripted=False,progress=None):
    if rules.get('respawn',False) or rules.get('prey_count',1)>1:
        from continuous_pursuit import continuous_episodes
        return continuous_episodes(predators,prey_genomes,ids,world,rules,seconds,pellet_count,grid,trace,scripted,progress)
    pos,pellets=make_arrangements(ids,world,pellet_count);n=len(ids);vel=np.zeros_like(pos)
    policies=[OrganismPolicy(side,pos[:,side],world,rules,grid) for side in range(2)]
    active=np.ones(n,bool);caught=np.zeros(n,bool);end=np.full(n,seconds,dtype=float);food=np.zeros(n);available=np.ones((n,pellet_count),bool)
    closest=np.linalg.norm(pos[:,0]-pos[:,1],axis=1);inside=np.zeros(n);avoid_time=np.zeros(n)
    collider=world[0];steps=int(round(seconds/DT))
    for tick in range(steps):
        if progress is not None and tick and tick%50==0:progress(tick*DT,seconds,int(active.sum()),n)
        if not active.any():break
        obs,seen=observe(pos,vel,(rules['predator_vision'],rules['prey_vision']),world[2],pellets,available,'unit')
        desired=np.stack([policies[side].command(genomes,pos[:,side],pos[:,1-side],obs[side],pellets,available,tick,active,scripted) for side,genomes in enumerate((predators,prey_genomes))],axis=1)
        seen=np.stack([p.seen for p in policies],axis=1)
        if trace is not None:
            trace.append(dict(positions=pos[0].tolist(),goals=[p.goals[0].tolist() for p in policies],modes=[str(p.modes[0]) if active[0] else 'captured' for p in policies],visible=seen[0].tolist(),visited=[np.flatnonzero(p.visited[0]).tolist() for p in policies],food=int(food[0]),remaining=pellets[0,available[0]].tolist(),caught=bool(caught[0]),capture_time=float(end[0]) if caught[0] else None))
        old=pos.copy();was_active=active.copy();near=active&(np.linalg.norm(old[:,0]-old[:,1],axis=1)<=2*collider.radius+2*rules['speed']*DT)
        hit_fraction=np.ones(n);tag=np.zeros(n,bool)
        for row in np.flatnonzero(active):
            paths=[[],[]] if near[row] else [None,None]
            for side in range(2):pos[row,side]=collider.move(pos[row,side],desired[row,side]*DT,path=paths[side])[0]
            if near[row]:
                fraction=capture_fraction(paths[0],paths[1],2*collider.radius)
                if fraction is not None:
                    tag[row]=True;hit_fraction[row]=fraction
                    for side in range(2):pos[row,side]=interpolate_path(paths[side],[fraction])[0]
        vel=(pos-old)/DT;closest=np.minimum(closest,np.linalg.norm(pos[:,0]-pos[:,1],axis=1))
        weights=was_active*DT*hit_fraction;inside+=(seen[:,0]*weights);avoid_time+=policies[1].avoiding*weights
        caught|=tag;end[tag]=(tick+hit_fraction[tag])*DT;active[tag]=False
        food+=collect_food(old[:,1],pos[:,1],pellets,available,was_active&~tag)
        for side in range(2):policies[side].mark(pos[:,side],tick+1,was_active)
    if trace is not None:
        seen=[bool(np.linalg.norm(pos[0,0]-pos[0,1])<=rules[key]) for key in ('predator_vision','prey_vision')]
        if rules.get('vision_occlusion',False):
            seen=[navigation(world).visible(pos[0,side],pos[0,1-side],rules[key]) for side,key in enumerate(('predator_vision','prey_vision'))]
        trace.append(dict(positions=pos[0].tolist(),goals=[p.goals[0].tolist() for p in policies],modes=[str(p.modes[0]) if active[0] else 'captured' for p in policies],visible=seen,visited=[np.flatnonzero(p.visited[0]).tolist() for p in policies],food=int(food[0]),remaining=pellets[0,available[0]].tolist(),caught=bool(caught[0]),capture_time=float(end[0]) if caught[0] else None))
    fraction=end/seconds;predator_score=caught*(2-fraction)+.2*np.exp(-closest/12);prey_score=fraction-caught+.25*food
    return np.c_[predator_score,prey_score,caught,end,food,inside/np.maximum(end,DT),avoid_time/np.maximum(end,DT),policies[0].visited.sum(axis=1),policies[1].visited.sum(axis=1)]

def render_replay(terrain,examples,rules,seconds,out):
    if rules.get('respawn',False) or rules.get('prey_count',1)>1:
        from continuous_pursuit import render_continuous
        return render_continuous(terrain,examples,rules,seconds,out)
    tile,gap,top=320,16,75;scale=tile/96;grid=8;font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',14)
    solid=np.array(terrain['solid']).reshape(96,96)
    sight=TerrainNavigation.__new__(TerrainNavigation)
    from grid_collision import GridCollider
    sight.sight=GridCollider(terrain,radius=0)
    bg=Image.fromarray(np.where(solid[:,:,None],np.array([108,119,116]),np.array([232,228,206])).astype('uint8')).resize((tile,tile),Image.Resampling.NEAREST).convert('RGBA')
    frames=[]
    for tick in range(0,int(round(seconds/DT))+1,2):
        im=Image.new('RGB',(len(examples)*tile+(len(examples)+1)*gap,top+tile+95),'#172321');draw=ImageDraw.Draw(im)
        draw.text((gap,7),f'Hybrid pursuit | coral predator vision {rules["predator_vision"]:g} | teal prey vision {rules["prey_vision"]:g} | playback 2x',font=font,fill='white')
        draw.text((gap,29),'Prey: avoid > visible food > explore | predator: pursue visible prey > explore',font=font,fill='white')
        draw.text((gap,50),'Green: prey visits | purple: route waypoint | vision clipped by terrain' if rules.get('vision_occlusion',False) else 'Green: prey visits | purple: route waypoint | circular vision',font=font,fill='#d6e4dc')
        for panel,trace in enumerate(examples):
            f=trace[min(tick,len(trace)-1)];image=bg.copy();x=gap+panel*(tile+gap)
            overlay=np.zeros((96,96,4),dtype='uint8')
            for cell in f['visited'][1]:
                gx,gy=cell%grid,cell//grid;overlay[gy*12:(gy+1)*12,gx*12:(gx+1)*12]=[50,160,85,75]
            overlay[solid.astype(bool),3]=0;image=Image.alpha_composite(image,Image.fromarray(overlay).resize((tile,tile),Image.Resampling.NEAREST));d=ImageDraw.Draw(image)
            for line in range(9):
                q=min(tile-1,round(line*tile/8));d.line((q,0,q,tile-1),fill='#9da995');d.line((0,q,tile-1,q),fill='#9da995')
            for point in f['remaining']:
                fx,fy=np.array(point)*scale;d.ellipse((fx-3,fy-3,fx+3,fy+3),fill='#e3b72f',outline='#66551a')
            for side,color,key in [(0,'#df755e','predator_vision'),(1,'#239b9d','prey_vision')]:
                px,py=np.array(f['positions'][side])*scale;r=rules[key]*scale
                if rules.get('vision_occlusion',False):
                    outline=[tuple(point*scale) for point in sight.vision_outline(f['positions'][side],rules[key])]
                    d.line(outline+[outline[0]],fill=color,width=1)
                else:d.ellipse((px-r,py-r,px+r,py+r),outline=color,width=1)
                if not f['caught']:
                    gx,gy=np.array(f['goals'][side])*scale;d.line((px,py,gx,gy),fill='#9a58ac',width=1)
                r=2.4*scale;d.ellipse((px-r,py-r,px+r,py+r),fill=color,outline='white' if f['visible'][side] else '#263330',width=2)
            im.paste(image.convert('RGB'),(x,top))
            lines=[f'{tick*DT:.1f}s | '+(f'captured at {f["capture_time"]:.2f}s' if f['caught'] else 'active')+f' | food {f["food"]}/12',f'Predator: {f["modes"][0]} | prey: {f["modes"][1]}',f'Prey detected by predator: {"yes" if f["visible"][0] else "no"}']
            for row,text in enumerate(lines):draw.text((x,top+tile+8+row*24),text,font=font,fill='#d6e4dc')
        frames.append(im)
    frames[0].save(out/'replay.gif',save_all=True,append_images=frames[1:],duration=100,loop=0)
    frames[min(25,len(frames)-1)].save(out/'replay-preview.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/hybrid-pursuit-pilot'))
    parser.add_argument('--arena',choices=('empty','terrain'),default='terrain')
    parser.add_argument('--terrain-map',type=Path,default=Path('results/landscape-previews/map-0-seed-42.json'))
    parser.add_argument('--generations',type=int,default=12);parser.add_argument('--seeds',type=int,default=3)
    parser.add_argument('--training-starts',type=int,default=4);parser.add_argument('--test-starts',type=int,default=24)
    parser.add_argument('--seconds',type=float,default=20);parser.add_argument('--replay-seconds',type=float,default=60)
    parser.add_argument('--no-pathfinding',action='store_true',help='Disable terrain routing for a matched comparison')
    parser.add_argument('--prey-count',type=int,default=1)
    parser.add_argument('--respawn',action='store_true',help='Continuous encounters with prey and food respawning')
    parser.add_argument('--memory-half-life',type=float,default=30,help='Visit-memory half-life in seconds in continuous mode')
    parser.add_argument('--no-prey-collisions',action='store_true',help='Continuous mode comparison: allow prey to pass through each other')
    parser.add_argument('--no-random-targets',action='store_true',help='Continuous mode comparison: use deterministic nearest targets')
    parser.add_argument('--food-spawn-mode',choices=('soft','strict'),default='soft')
    parser.add_argument('--food-spawn-scale',type=float,default=12.,help='Distance scale for smooth food spawn avoidance (cells)')
    parser.add_argument('--no-occlusion',action='store_true',help='Use circular vision through terrain')
    parser.add_argument('--eval-every',type=int,default=3)
    parser.add_argument('--predator-vision',type=float,default=RULES['predator_vision']);parser.add_argument('--prey-vision',type=float,default=RULES['prey_vision'])
    args=parser.parse_args()
    if args.prey_count<1 or args.memory_half_life<=0 or args.food_spawn_scale<=0:parser.error('Prey count, memory half-life and food spawn scale must be positive')
    continuous=args.respawn or args.prey_count>1
    if continuous and not args.respawn:parser.error('Multiple prey currently require --respawn')
    if min(args.generations,args.seeds,args.training_starts,args.test_starts,args.seconds,args.replay_seconds,args.eval_every,args.predator_vision,args.prey_vision)<=0:parser.error('All budgets and ranges must be positive')
    if args.training_starts>100 or args.test_starts>1000:parser.error('Use at most 100 training starts and 1000 test starts')
    for duration in (args.seconds,args.replay_seconds):
        if abs(round(duration/DT)*DT-duration)>1e-8:parser.error('Durations must be multiples of the timestep')
    args.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    rules=dict(RULES,predator_vision=args.predator_vision,prey_vision=args.prey_vision,pathfinding=not args.no_pathfinding,vision_occlusion=not args.no_occlusion)
    if continuous:
        rules.update(prey_count=args.prey_count,respawn=True,memory_half_life=args.memory_half_life,respawn_delay=1.,prey_collisions=not args.no_prey_collisions,random_prey_targets=not args.no_random_targets,target_commitment_seconds=3.,food_spawn_mode=args.food_spawn_mode,food_spawn_scale=args.food_spawn_scale,
                     capture='Contact removes that prey until respawn; fixed-duration encounter and predator motion continue. Captured prey cannot collect food in that timestep.',
                     memory='Private visit confidence decays exponentially, becoming unvisited below 0.5. New individuals start with fresh memory. Static terrain remains known.',
                     population_rule='One predator selects nearest visible active prey. All prey share an evolved genome but private states and independent seeded target-choice streams. Body collisions are controlled by prey_collisions.',
                     target_choice_rule='When enabled, independently choose among three closest visible eligible pellets or five closest eligible unvisited exploration tiles, with distance preference. Keep food choices for up to three seconds; keep exploration goals until reached or interrupted. No shared target assignments or hidden food information.')
    if continuous:
        rules['respawn_rule']='Prey respawn outside predator vision. Food uses saved food_spawn_mode (soft distance bias or strict outside-prey-vision). All spawns avoid terrain/body overlap; food spacing is six cells. Retry failed placements after one second.'
        rules['food_spawn_rule']='Soft site weight = .05 + .95*(1-exp(-.5*(distance_to_nearest_active_prey/food_spawn_scale)^2)); no hard vision cutoff. Strict mode retains the original radius exclusion.'
    metric_names=('captures','alive_fraction','food_per_prey','inside_predator_vision','avoid_fraction','predator_tiles','prey_tiles') if continuous else ('capture','duration','food','inside_predator_vision','avoid_fraction','predator_tiles','prey_tiles')
    terrain=json.loads(args.terrain_map.read_text()) if args.arena=='terrain' else dict(resolution=96,solid=[0]*96**2,preset='Empty arena')
    world=setup(terrain);rows=[];history={};champion_data={};reference=[]
    def progress(label):
        return lambda elapsed,total,active,count: print(f'{label} | simulated {elapsed:g}/{total:g}s | active {active}/{count} | elapsed {time.perf_counter()-started:.1f}s',flush=True)
    (args.output/'terrain.json').write_text(json.dumps(terrain));(args.output/'instructions.json').write_text(json.dumps(dict(pursuit=rules,foraging_and_recovery=HYBRID_INSTRUCTIONS),indent=2))
    for seed in range(args.seeds):
        initial=np.random.default_rng(710+seed).normal(0,.45,(4,GENES));populations=[initial.copy(),initial.copy()];champions=[initial[0].copy(),initial[0].copy()]
        streams=[np.random.default_rng(1200+2*seed+side) for side in range(2)]
        test_ids=np.arange(args.test_starts)+90000000+seed*1000
        for generation in range(args.generations+1):
            if generation:
                ids=np.arange(args.training_starts)+1000000+generation*10000+seed*100
                a=np.repeat(populations[0],4*len(ids),axis=0);b=np.tile(np.repeat(populations[1],len(ids),axis=0),(4,1))
                measured=episodes(a,b,np.tile(ids,16),world,rules,args.seconds,progress=progress(f'Train seed {seed+1}/{args.seeds} gen {generation}/{args.generations}')).reshape(4,4,len(ids),9)
                scores=[measured[:,:,:,0].mean(axis=(1,2)),measured[:,:,:,1].mean(axis=(0,2))]
                champions=[population[score.argmax()].copy() for population,score in zip(populations,scores)]
                populations=[]
                for genome,rng in zip(champions,streams):
                    children=[genome.copy()]
                    for _ in range(3):
                        child=genome.copy();idx=rng.choice(GENES,12,replace=False);child[idx]+=rng.normal(0,.35,12);children.append(child)
                    populations.append(np.array(children))
            if generation%args.eval_every==0 or generation==args.generations:
                history[f'{seed}/{generation}']=[g.tolist() for g in champions];(args.output/'history.json').write_text(json.dumps(history))
                result=episodes(np.repeat(champions[0][None],len(test_ids),axis=0),np.repeat(champions[1][None],len(test_ids),axis=0),test_ids,world,rules,args.seconds,progress=progress(f'Benchmark seed {seed+1}/{args.seeds} gen {generation}'))
                for trial,row in enumerate(result):rows.append(dict(seed=seed,generation=generation,start=int(test_ids[trial]),**dict(zip(metric_names,map(float,row[2:])))))
                with (args.output/'measurements.csv').open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                print(f'Benchmark | seed {seed+1}/{args.seeds} | gen {generation} | {metric_names[0]} {result[:,2].mean():.3f} | {metric_names[2]} {result[:,4].mean():.2f} | prey detected {result[:,5].mean():.1%}',flush=True)
            print(f'Hybrid pursuit | seed {seed+1}/{args.seeds} | generation {generation}/{args.generations} | {time.perf_counter()-started:.1f}s',flush=True)
        champion_data[str(seed)]=[g.tolist() for g in champions];(args.output/'champions.json').write_text(json.dumps(champion_data))
        scripted=episodes(np.repeat(champions[0][None],len(test_ids),axis=0),np.repeat(champions[1][None],len(test_ids),axis=0),test_ids,world,rules,args.seconds,scripted=True)
        reference.append(dict(seed=seed,**{metric_names[0]:float(scripted[:,2].mean()),metric_names[2]:float(scripted[:,4].mean())}))
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained');summary={}
    for ax,key,title in zip(axes.flat,[metric_names[0],metric_names[2],'inside_predator_vision','prey_tiles'],['Predator captures per encounter' if continuous else 'Predator capture fraction','Prey food per individual' if continuous else 'Prey food collection','Fraction of alive time detected by predator','Prey recently visited tiles' if continuous else 'Prey grid tiles visited']):
        gs=sorted(set(r['generation'] for r in rows));values=np.array([[np.mean([r[key] for r in rows if r['seed']==seed and r['generation']==g]) for g in gs] for seed in range(args.seeds)])
        mean,low,high=seed_mean_interval(values);ax.plot(gs,mean,color='#188b87',lw=2);ax.fill_between(gs,low,high,color='#188b87',alpha=.2)
        ax.set(title=title,xlabel='Generation',ylabel='Fraction' if key in ('capture','inside_predator_vision') else 'Captures' if key=='captures' else 'Pellets' if key in ('food','food_per_prey') else 'Tiles',ylim=(0,1 if key in ('capture','inside_predator_vision') else None))
        summary[key]=dict(initial=float(mean[0]),final=float(mean[-1]),ci95=[float(low[-1]),float(high[-1])],seed_means=values[:,-1].tolist())
    fig.suptitle(f'Explicit hybrid predator/prey co-evolution | {args.seeds} seeds, mean and pointwise 95% CI\nVision: predator {args.predator_vision:g}, prey {args.prey_vision:g}; programmed priorities, evolved movement')
    fig.savefig(args.output/'outcomes.png',dpi=170);plt.close(fig)
    examples=[]
    for start in (90000000,90000001,90000002):
        trace=[];a,b=np.array(champion_data['0']);episodes(a[None],b[None],[start],world,rules,args.replay_seconds,trace=trace);examples.append(trace)
    render_replay(terrain,examples,rules,args.replay_seconds,args.output);(args.output/'replays.json').write_text(json.dumps(examples))
    summary['scripted_reference']=reference;(args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    fitness={'predator':'captures + .2*time_mean(exp(-nearest_active_prey_distance/12))','prey':'alive_seconds/(prey_count*episode_seconds) - captures/prey_count + .25*food_collected/prey_count'} if continuous else {'predator':'capture*(2-alive_fraction) + .2*exp(-closest_distance/12)','prey':'alive_fraction - capture + .25*pellets'}
    notes=['Only movement NN weights evolve; target rules, terrain routing, vision and recovery are programmed.','Current-opponent curves describe a changing matchup, not absolute progress.','Saved replays use the first seed and first three test starts.']
    if continuous:
        notes += ['One predator; all prey share one genome but have independent memories and recovery state. Prey collisions and random target choices follow their saved rule flags.','Fixed-duration trials continue through captures. Prey and food retry respawning after one second if no legal site exists.','Prey spawn strictly outside predator vision radius. Food placement follows the saved food_spawn_mode: strict excludes active prey vision radii; soft weights legal sites by .05 + .95*(1-exp(-.5*(nearest_prey_distance/food_spawn_scale)^2)). All spawns are terrain-clear and avoid body overlap; food spacing is at least six cells.','Predator selects nearest visible active prey. Individual prey retain avoidance, food and exploration priorities.','Visit confidence decays exponentially; below 0.5 a tile becomes unvisited for planning. Lifetime coverage is not the recently visited tile count. Respawned individuals have fresh memory.','Capture is continuous along collision-resolved paths; captured prey cannot collect food in that timestep. Predator motion continues. Enabled prey collisions use simultaneous swept circles with equal-mass inelastic sliding, removing captured bodies at contact. Each pellet is awarded to the earliest eligible prey swept pickup, ties by prey index.','Capture metric is a count, food metric is total per prey slot over the fixed trial, alive_fraction excludes respawn downtime. These differ from the original single-life experiment.','Replay shows the first encounter, with prey 1 memory fading. All three encounters are saved as JSON.']
    else:
        notes += ['Capture ends the encounter without respawn.','Each organism keeps private visit memory; terrain knowledge is static.','Detection respects saved vision occlusion setting. No hidden opponent position or relative velocity is used.']
    (args.output/'design.json').write_text(json.dumps(dict(generations=args.generations,seeds=args.seeds,population_per_role=4,genes_per_role=GENES,training_starts=args.training_starts,test_starts=args.test_starts,episode_seconds=args.seconds,replay_seconds=args.replay_seconds,arena=args.arena,rules=rules,fitness=fitness,notes=notes,seconds=time.perf_counter()-started),indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
