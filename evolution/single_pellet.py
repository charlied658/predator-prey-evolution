"""Minimal foraging diagnostic using the pursuit controller and collision code."""
import argparse,csv,json,time,hashlib,inspect,shutil
from pathlib import Path
import numpy as np
import vision_pilot as pursuit
from grid_collision import GridCollider
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
from vision_pilot import (GENES,INPUTS,DT,STEPS,SPEED,FOOD_RADIUS,observe,network,
                          collect_food,setup,seed_mean_interval)


REC_HIDDEN=6
REC_GENES=(INPUTS+1)*REC_HIDDEN+REC_HIDDEN**2+(REC_HIDDEN+1)*2


def recurrent_network(genomes,obs,state):
    """Elman recurrent controller; state belongs to one evaluation batch only."""
    cut=(INPUTS+1)*REC_HIDDEN;end=cut+REC_HIDDEN**2
    incoming=genomes[:,:cut].reshape(-1,INPUTS+1,REC_HIDDEN)
    recurrent=genomes[:,cut:end].reshape(-1,REC_HIDDEN,REC_HIDDEN)
    outgoing=genomes[:,end:].reshape(-1,REC_HIDDEN+1,2)
    updated=np.tanh(np.einsum('ni,nij->nj',np.c_[obs,np.ones(len(obs))],incoming)+np.einsum('ni,nij->nj',state,recurrent))
    command=np.tanh(np.einsum('ni,nij->nj',np.c_[updated,np.ones(len(obs))],outgoing))
    return command,updated


def map_gene_count(grid_cells=8):
    return (INPUTS+2+grid_cells**2+1)*8+18


def map_observation(obs,position,visited,blank=False):
    """Only past centre visits and current position; never unseen food."""
    flags=np.zeros_like(visited,dtype=float) if blank else visited.astype(float)
    return np.c_[obs,position/48-1,flags]


def map_network(genomes,obs):
    cut=(obs.shape[1]+1)*8
    incoming=genomes[:,:cut].reshape(-1,obs.shape[1]+1,8)
    outgoing=genomes[:,cut:].reshape(-1,9,2)
    hidden=np.tanh(np.einsum('ni,nij->nj',np.c_[obs,np.ones(len(obs))],incoming))
    return np.tanh(np.einsum('ni,nij->nj',np.c_[hidden,np.ones(len(obs))],outgoing))


HYBRID_INSTRUCTIONS=json.loads(Path(__file__).with_name('hybrid_instructions.json').read_text())


def hybrid_exploration_sites(world,grid_cells=8):
    """One body-accessible representative per grid tile; no route assistance."""
    sites=world[1];centres=np.array([((x+.5)*96/grid_cells,(y+.5)*96/grid_cells) for y in range(grid_cells) for x in range(grid_cells)])
    cells=np.clip((sites*grid_cells/96).astype(int),0,grid_cells-1);ids=cells[:,1]*grid_cells+cells[:,0]
    eligible=np.zeros(grid_cells**2,bool)
    for cell in range(grid_cells**2):
        candidates=sites[ids==cell]
        if len(candidates):
            eligible[cell]=True;centres[cell]=candidates[np.linalg.norm(candidates-centres[cell],axis=1).argmin()]
    return centres,eligible


def hybrid_target(position,pellets,available,visited,last_visit,exploration_target,grid_cells=8,exploration_sites=None,blocked_tiles=None,food_vision=24):
    """Programmed priorities; only visible food may influence target choice."""
    count=len(position);rows=np.arange(count)
    delta=pellets-position[:,None,:];distance=np.linalg.norm(delta,axis=-1)
    visible=available & (distance<=food_vision)
    food_index=np.where(visible,distance,np.inf).argmin(axis=1);foraging=visible.any(axis=1)
    centres=np.array([((x+.5)*96/grid_cells,(y+.5)*96/grid_cells) for y in range(grid_cells) for x in range(grid_cells)])
    eligible=np.ones(grid_cells**2,bool)
    if exploration_sites is not None:centres,eligible=exploration_sites
    previous=centres[np.maximum(exploration_target,0)]
    reached=np.linalg.norm(previous-position,axis=1)<=HYBRID_INSTRUCTIONS['goal_reached_distance']
    choose=~foraging & ((exploration_target<0)|reached)
    distances=np.linalg.norm(centres[None]-position[:,None],axis=-1)
    blocked=np.zeros_like(visited) if blocked_tiles is None else blocked_tiles
    selectable=eligible[None]&~blocked
    exhausted=~selectable.any(axis=1);selectable[exhausted]=eligible
    distances[visited|~selectable]=np.inf
    candidate=distances.argmin(axis=1)
    all_visited=(visited|~selectable).all(axis=1);candidate[all_visited]=np.where(selectable[all_visited],last_visit[all_visited],np.inf).argmin(axis=1)
    exploration_target[choose]=candidate[choose]
    exploration_target[foraging]=-1
    goals=pellets[rows,food_index].copy()
    goals[~foraging]=centres[exploration_target[~foraging]]
    return goals,foraging


def hybrid_escape_waypoint(collider,position,blocked_goal):
    """Local collision-clear backoff, not global route planning."""
    length=HYBRID_INSTRUCTIONS['escape_distance'];toward=blocked_goal-position
    toward=toward/max(np.linalg.norm(toward),1e-12);best=-np.inf;goal=position.copy()
    for angle in np.arange(HYBRID_INSTRUCTIONS['escape_directions'])*2*np.pi/HYBRID_INSTRUCTIONS['escape_directions']:
        direction=np.array([np.cos(angle),np.sin(angle)])
        fraction,_=collider.sweep(position,direction*length)
        clearance=max(0,length*fraction-1e-4)
        score=clearance-.25*np.dot(direction,toward)
        if clearance>.25 and score>best:
            best=score;goal=position+direction*clearance
    return goal


def hybrid_observation(obs,position,goals,foraging):
    """Reuse the 170-weight NN: target direction plus explicit task mode."""
    result=obs.copy();delta=goals-position;distance=np.linalg.norm(delta,axis=1)
    result[:,:5]=0;result[:,0]=foraging;result[:,1]=distance/96
    result[:,-3:-1]=delta/np.maximum(distance[:,None],1e-12);result[:,-1]=distance>1e-12
    return result


def hybrid_command(raw,position,goals):
    """Enforce progress toward the instructed target in an obstacle-free arena."""
    delta=goals-position;direction=delta/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-12)
    angle=np.clip(raw[:,0],-1,1)*np.deg2rad(HYBRID_INSTRUCTIONS['steering_limit_degrees'])
    cosine,sine=np.cos(angle),np.sin(angle)
    heading=np.c_[direction[:,0]*cosine-direction[:,1]*sine,direction[:,0]*sine+direction[:,1]*cosine]
    minimum=HYBRID_INSTRUCTIONS['minimum_throttle'];throttle=minimum+(1-minimum)*(np.clip(raw[:,1],-1,1)+1)/2
    return heading*throttle[:,None]


def arrangements(ids,pellet_count=1,sites=None):
    """Nested initially visible pellet sets, spaced at least six cells apart."""
    positions=[];pellets=[]
    for seed in ids:
        rng=np.random.default_rng(int(seed))
        p=rng.uniform(28,68,2) if sites is None else sites[rng.integers(len(sites))].copy()
        food=[]
        for _ in range(10000):
            if sites is None:
                angle=rng.uniform(0,2*np.pi);distance=rng.uniform(6,20)
                point=p+distance*np.array([np.cos(angle),np.sin(angle)])
            else:
                point=sites[rng.integers(len(sites))]
                if np.linalg.norm(point-p)<=FOOD_RADIUS+2:continue
            if all(np.linalg.norm(point-other)>=6 for other in food):food.append(point)
            if len(food)==pellet_count:break
        if len(food)!=pellet_count:raise ValueError('Could not place requested pellets')
        positions.append(p);pellets.append(food)
    return np.array(positions),np.array(pellets)


def evaluate(genomes,ids,world,policy='network',trace=None,pellet_count=1,food_layout='nearby',grid_cells=8,controller='feedforward',reset_memory=False,blank_map=False,duration_seconds=None,recovery=None,progress=None):
    episode_steps=STEPS if duration_seconds is None else int(round(duration_seconds/DT))
    if episode_steps<1:raise ValueError('Encounter duration must be positive')
    collider,sites,rays=world
    position,pellets=arrangements(ids,pellet_count) if food_layout=='nearby' else arrangements(ids,pellet_count,sites)
    exploration_sites=hybrid_exploration_sites(world,grid_cells) if controller=='hybrid' and any(collider.solid) else None
    recovery_on=controller=='hybrid' and (HYBRID_INSTRUCTIONS.get('recovery_enabled',False) if recovery is None else recovery)
    recovery_window=max(1,round(HYBRID_INSTRUCTIONS.get('stuck_seconds',2)/DT))
    history=np.repeat(position[:,None,:],recovery_window+1,axis=1)
    recover_until=np.zeros(len(ids),int);next_check=np.full(len(ids),recovery_window,int)
    recovery_goal=position.copy();recovery_count=np.zeros(len(ids),int)
    food_cooldown=np.zeros((len(ids),pellet_count),int);tile_cooldown=np.zeros((len(ids),grid_cells*grid_cells),int)
    velocity=np.zeros_like(position)
    exploration_target=np.full(len(ids),-1,dtype=int);goals=position.copy();foraging=np.zeros(len(ids),bool)
    last_visit=np.full((len(ids),grid_cells*grid_cells),-1,dtype=int)
    state=np.zeros((len(ids),REC_HIDDEN))
    visited=np.zeros((len(ids),grid_cells*grid_cells),bool)
    def mark_visited(tick=0):
        cells=np.clip((position*grid_cells/96).astype(int),0,grid_cells-1)
        visited[np.arange(len(ids)),cells[:,1]*grid_cells+cells[:,0]]=True
        last_visit[np.arange(len(ids)),cells[:,1]*grid_cells+cells[:,0]]=tick
    mark_visited()
    active=np.ones(len(ids),bool);available=np.ones((len(ids),pellet_count),bool);collected=np.zeros(len(ids));success=np.zeros(len(ids),bool);end=np.full(len(ids),episode_steps*DT)
    contact_totals=np.zeros(len(ids),int);last_contacts=np.zeros(len(ids),int)
    elapsed_steps=0
    def record_progress(step):
        if progress is not None:
            eligible=exploration_sites[1] if exploration_sites is not None else np.ones(grid_cells**2,bool)
            progress.append(dict(seconds=step*DT,food=collected.copy(),visited=visited.sum(axis=1),reachable_visited=visited[:,eligible].sum(axis=1)))
    for tick in range(episode_steps):
        if tick%10==0:record_progress(tick)
        if not active.any():break
        # Dummy opponent has no effect: all five opponent-specific inputs are
        # masked. Other sensors and NN layout exactly match the pursuit pilot.
        pos=np.stack((position,position),axis=1);vel=np.stack((np.zeros_like(velocity),velocity),axis=1)
        inputs,_=observe(pos,vel,(24,24),rays,pellets,available,'unit');obs=inputs[1];obs[:,:5]=0
        recovering=np.zeros(len(ids),bool)
        if controller=='hybrid':
            if recovery_on:
                reached_escape=np.linalg.norm(recovery_goal-position,axis=1)<=HYBRID_INSTRUCTIONS['goal_reached_distance']
                recover_until[reached_escape & (recover_until>tick)]=tick
            planning_food=available & (food_cooldown<=tick) if recovery_on else available
            goals,foraging=hybrid_target(position,pellets,planning_food,visited,last_visit,exploration_target,grid_cells,exploration_sites,tile_cooldown>tick if recovery_on else None)
            if recovery_on and tick>=recovery_window:
                confined=np.linalg.norm(history-position[:,None],axis=-1).max(axis=1)<HYBRID_INSTRUCTIONS['stuck_radius']
                stuck=active & confined & (tick>=next_check) & (tick>=recover_until)
                for row in np.flatnonzero(stuck):
                    until=tick+round(HYBRID_INSTRUCTIONS['target_cooldown_seconds']/DT)
                    if foraging[row]:
                        nearest=np.linalg.norm(pellets[row]-goals[row],axis=1).argmin();food_cooldown[row,nearest]=until
                    elif exploration_target[row]>=0:tile_cooldown[row,exploration_target[row]]=until
                    exploration_target[row]=-1
                    recovery_goal[row]=hybrid_escape_waypoint(collider,position[row],goals[row])
                    recover_until[row]=tick+round(HYBRID_INSTRUCTIONS['recovery_seconds']/DT)
                    next_check[row]=recover_until[row]+recovery_window;recovery_count[row]+=1
                recovering=active & (tick<recover_until)
                goals[recovering]=recovery_goal[recovering];foraging[recovering]=False
            obs=hybrid_observation(obs,position,goals,foraging)
        if trace is not None:trace.append(dict(position=position[0].tolist(),pellet=pellets[0,0].tolist(),collected=bool(success[0]),remaining=pellets[0,available[0]].tolist(),count=int(collected[0]),visited=np.flatnonzero(visited[0]).tolist(),target=goals[0].tolist() if controller=='hybrid' else None,mode=('recover' if recovering[0] else 'forage' if foraging[0] else 'explore') if controller=='hybrid' else None,recoveries=int(recovery_count[0]),contacts=int(contact_totals[0]),last_contacts=int(last_contacts[0])))
        if policy=='direct':command=obs[:,-3:-1]
        elif controller=='hybrid':command=hybrid_command(network(genomes,obs),position,goals)
        elif controller in ('visited-map','position-only'):
            mapped=map_observation(obs,position,visited,blank=blank_map or controller=='position-only')
            command=map_network(genomes,mapped)
        elif controller=='recurrent':
            if reset_memory:state[:]=0
            command,state=recurrent_network(genomes,obs,state)
        else:command=network(genomes,obs)
        if recovering.any():
            delta=goals[recovering]-position[recovering]
            command[recovering]=delta/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-12)*HYBRID_INSTRUCTIONS['recovery_throttle']
        desired=command*SPEED/np.maximum(1,np.linalg.norm(command,axis=1,keepdims=True))
        old=position.copy()
        last_contacts[:]=0
        for row in np.flatnonzero(active):
            position[row],contacts=collider.move(position[row],desired[row]*DT)
            last_contacts[row]=contacts;contact_totals[row]+=contacts
        velocity=(position-old)/DT
        mark_visited(tick+1)
        history[:,(tick+1)%(recovery_window+1)]=position
        collected+=collect_food(old,position,pellets,available,active)
        finished=active & ~available.any(axis=1)
        success|=finished;end[finished]=(tick+1)*DT;active[finished]=False
        elapsed_steps=tick+1
    if progress is not None and (not progress or progress[-1]['seconds']<elapsed_steps*DT):record_progress(elapsed_steps)
    if trace is not None:trace.append(dict(position=position[0].tolist(),pellet=pellets[0,0].tolist(),collected=bool(success[0]),remaining=pellets[0,available[0]].tolist(),count=int(collected[0]),visited=np.flatnonzero(visited[0]).tolist(),contacts=int(contact_totals[0]),last_contacts=int(last_contacts[0])))
    if trace is not None and controller=='hybrid':
        trace[-1]['target']=None if success[0] else goals[0].tolist()
        trace[-1]['mode']='complete' if success[0] else 'recover' if recovering[0] else 'forage' if foraging[0] else 'explore'
        trace[-1]['recoveries']=int(recovery_count[0])
    return np.c_[success,end,collected,visited.sum(axis=1)]


def replay(world,genome,out,pellet_count=1,food_layout='nearby',grid_cells=8,exploration_reward=0.,controller='feedforward'):
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',15);tile=280;gap=16;header=64;examples=[]
    for start in (90000000,90000001,90000002):
        trace=[];evaluate(genome[None],[start],world,trace=trace,pellet_count=pellet_count,food_layout=food_layout,grid_cells=grid_cells,controller=controller);examples.append(trace)
    solid=np.array(world[0].solid).reshape(96,96)
    pixels=np.where(solid[:,:,None],np.array([108,119,116]),np.array([232,228,206])).astype('uint8')
    background=Image.fromarray(pixels).resize((tile,tile),Image.Resampling.NEAREST).convert('RGBA')
    frames=[]
    for tick in range(STEPS+1):
        im=Image.new('RGB',(3*tile+4*gap,header+tile+40),'#172321');d=ImageDraw.Draw(im)
        d.text((gap,8),f'{controller} | {pellet_count} pellets | bonus {exploration_reward:g} | final seed 1 | first 3 starts',font=font,fill='white')
        d.text((gap,32),'Green: visited | cream: unvisited | gray: terrain | teal: prey | yellow: food | circle: vision',font=font,fill='white')
        for i,trace in enumerate(examples):
            x=gap+i*(tile+gap);panel=background.copy();pd=ImageDraw.Draw(panel);f=trace[min(tick,len(trace)-1)]
            overlay=np.zeros((96,96,4),dtype='uint8')
            cell_size=96//grid_cells
            for cell in f['visited']:
                gx,gy=cell%grid_cells,cell//grid_cells
                overlay[gy*cell_size:(gy+1)*cell_size,gx*cell_size:(gx+1)*cell_size]=[50,160,85,90]
            overlay[solid.astype(bool),3]=0
            panel=Image.alpha_composite(panel,Image.fromarray(overlay).resize((tile,tile),Image.Resampling.NEAREST));pd=ImageDraw.Draw(panel)
            for line in range(grid_cells+1):
                q=min(tile-1,round(line*tile/grid_cells));pd.line((q,0,q,tile-1),fill='#9da995',width=1);pd.line((0,q,tile-1,q),fill='#9da995',width=1)
            px,py=np.array(f['position'])*tile/96;radius=24*tile/96
            pd.ellipse((px-radius,py-radius,px+radius,py+radius),outline='#66a7a6',width=1)
            points=[tuple(np.array(fr['position'])*tile/96) for fr in trace[max(0,min(tick,len(trace)-1)-30):min(tick+1,len(trace))]]
            if len(points)>1:pd.line(points,fill='#86b8b0',width=2)
            if f.get('target') is not None:
                tx,ty=np.array(f['target'])*tile/96
                pd.line((px,py,tx,ty),fill='#9a58ac',width=1)
                pd.ellipse((tx-4,ty-4,tx+4,ty+4),outline='#9a58ac',width=2)
            for point in f['remaining']:
                fx,fy=np.array(point)*tile/96;pd.ellipse((fx-3,fy-3,fx+3,fy+3),fill='#e3b72f',outline='#66551a')
            r=2.4*tile/96;pd.ellipse((px-r,py-r,px+r,py+r),fill='#239b9d',outline='white',width=2)
            im.paste(panel.convert('RGB'),(x,header));d.text((x,header+tile+8),f'{tick*DT:.1f}s | food {f["count"]}/{pellet_count} | '+(f.get('mode') or f'tiles {len(f["visited"])}'),font=font,fill='white')
        frames.append(im)
    frames[0].save(out/'replay.gif',save_all=True,append_images=frames[1:],duration=100,loop=0)
    frames[30].save(out/'replay-preview.png')


def run_compatibility(args,count,world,controller,genes,mutation_count):
    """Conservative cache key: settings, world contents and executable model code."""
    settings=dict(pellets=count,generations=args.generations,seeds=args.seeds,controller=controller,genes=genes,food_layout=args.food_layout,grid_cells=getattr(args,'grid_cells',8),exploration_reward=getattr(args,'exploration_reward',0.),population=4,training_starts=8,test_starts=128,mutation_count=mutation_count,mutation_sigma=.35,vision=24,dt=DT,steps=STEPS,speed=SPEED,food_radius=FOOD_RADIUS,body_radius=world[0].radius)
    digest=hashlib.sha256()
    digest.update(np.array(world[0].solid,dtype=np.uint8).tobytes())
    for array in world[1:]:digest.update(np.ascontiguousarray(array,dtype='<f8').tobytes())
    functions=[arrangements,evaluate,network,observe,collect_food,map_observation,map_network,recurrent_network,pursuit.food_observation,GridCollider,run_count]
    if controller=='hybrid':functions += [hybrid_exploration_sites,hybrid_target,hybrid_escape_waypoint,hybrid_observation,hybrid_command]
    code='\n'.join(inspect.getsource(fn) for fn in functions)
    if controller=='hybrid':settings['instructions']=HYBRID_INSTRUCTIONS
    payload=dict(settings=settings,world_sha256=digest.hexdigest(),code_sha256=hashlib.sha256(code.encode()).hexdigest())
    payload['fingerprint']=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    return payload


def load_run_rows(source,args):
    with (source/'measurements.csv').open() as f:raw=list(csv.DictReader(f))
    required={'seed','generation','food','visited','success','capped_time','fraction','fitness'}
    if not raw or not required.issubset(raw[0]):raise ValueError('Cached run lacks required measurements')
    rows=[{k:int(v) if k in ('seed','generation') else float(v) for k,v in r.items()} for r in raw]
    checkpoints=set(range(0,args.generations+1,5))|{args.generations}
    expected={(seed,g) for seed in range(args.seeds) for g in checkpoints}
    actual=[(r['seed'],r['generation']) for r in rows]
    if set(actual)!=expected or len(actual)!=len(expected):raise ValueError('Cached run is incomplete or has different checkpoints/seeds')
    if not all(np.isfinite(v) for r in rows for v in r.values()):raise ValueError('Cached run contains nonfinite measurements')
    return rows


def reuse_run(source,out,args,count,world,controller,genes,compatibility,explicit=False):
    source=source.resolve();manifest=source/'compatibility.json'
    cached=json.loads(manifest.read_text()) if manifest.exists() else None
    if cached is not None and cached.get('fingerprint')!=compatibility['fingerprint']:
        raise ValueError('Cached control settings, world, or simulation/training code differ')
    if cached is None:
        if not explicit:raise ValueError('Unverified legacy run')
        design=json.loads((source/'design.json').read_text())
        for key in ('pellets','generations','seeds','controller','genes','food_layout','grid_cells','exploration_reward','population','training_starts','test_starts','mutation_count','mutation_sigma','vision'):
            if design.get(key)!=compatibility['settings'][key]:raise ValueError(f'Cached control mismatch: {key}')
        if design.get('terrain')!=bool(any(world[0].solid)):raise ValueError('Cached arena differs')
    rows=load_run_rows(source,args)
    champions=np.array(json.loads((source/'champions.json').read_text()))
    if champions.shape!=(args.seeds,genes):raise ValueError('Cached champion dimensions differ')
    references=json.loads((source/'reference.json').read_text()) if (source/'reference.json').exists() else []
    if cached is None:
        print('Verifying legacy control: initial/final scores on all held-out starts (no evolution)',flush=True)
        for seed in range(args.seeds):
            ids=np.arange(128)+90000000+seed*1000
            if not (source/'reference.json').exists():
                direct=evaluate(np.zeros((128,genes)),ids,world,policy='direct',pellet_count=count,food_layout=args.food_layout,grid_cells=args.grid_cells,controller=controller)
                references.append(float(direct[:,2].mean()))
            initial=np.random.default_rng(710+seed).normal(0,.45,(4,genes))[0]
            for generation,genome in [(0,initial),(args.generations,champions[seed])]:
                measured=evaluate(np.repeat(genome[None],128,axis=0),ids,world,pellet_count=count,food_layout=args.food_layout,grid_cells=args.grid_cells,controller=controller)
                old=next(r for r in rows if r['seed']==seed and r['generation']==generation)
                expected=[old['success'],old['capped_time'],old['food'],old['visited']]
                np.testing.assert_allclose(measured.mean(axis=0),expected,rtol=0,atol=1e-10,err_msg='Legacy benchmark no longer reproduces')
    if cached and 'artifact_hashes' in cached:
        for filename,expected_hash in cached['artifact_hashes'].items():
            if hashlib.sha256((source/filename).read_bytes()).hexdigest()!=expected_hash:raise ValueError(f'Cached artifact changed: {filename}')
    out.mkdir(exist_ok=False)
    for filename in ('measurements.csv','measurements-reset.csv','champions.json','design.json','reference.json','replay.gif','replay-preview.png'):
        if (source/filename).exists():shutil.copy2(source/filename,out/filename)
    (out/'reference.json').write_text(json.dumps(references))
    copied=dict(compatibility)
    copied['artifact_hashes']={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ('measurements.csv','champions.json','design.json','reference.json')}
    if (out/'measurements-reset.csv').exists():copied['artifact_hashes']['measurements-reset.csv']=hashlib.sha256((out/'measurements-reset.csv').read_bytes()).hexdigest()
    (out/'compatibility.json').write_text(json.dumps(copied,indent=2))
    registry=Path(__file__).resolve().parent.parent/'results'/'.foraging-cache';registry.mkdir(parents=True,exist_ok=True)
    index=registry/f'{compatibility["fingerprint"]}.json';temp=index.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(source=str(out.resolve()))));temp.replace(index)
    (out/'cache-reuse.json').write_text(json.dumps(dict(source=str(source),training_skipped=True,legacy_endpoint_verification=cached is None,notes=['Original result files are copied without modifying the source. Legacy endpoint checks validate compatibility, not original training provenance.']),indent=2))
    print(f'REUSED completed {controller} run: {source} (training skipped)',flush=True)
    return rows,references


def run_count(args,count,out,world):
    weight=getattr(args,'exploration_reward',0.);grid_cells=getattr(args,'grid_cells',8)
    controller=getattr(args,'controller','feedforward');genes=REC_GENES if controller=='recurrent' else map_gene_count(grid_cells) if controller in ('visited-map','position-only') else GENES
    mutation_count=max(1,round(genes*12/GENES)) if controller in ('visited-map','position-only') else 12
    reset_rows=[]
    compatibility=run_compatibility(args,count,world,controller,genes,mutation_count)
    registry=Path(__file__).resolve().parent.parent/'results'/'.foraging-cache'
    cache_index=registry/f'{compatibility["fingerprint"]}.json'
    requested=getattr(args,'reuse_control',None) if controller=='position-only' else None
    if requested:
        return reuse_run(requested,out,args,count,world,controller,genes,compatibility,explicit=True)
    if cache_index.exists():
        try:
            source=Path(json.loads(cache_index.read_text())['source'])
            return reuse_run(source,out,args,count,world,controller,genes,compatibility)
        except (ValueError,OSError,KeyError,AssertionError) as error:
            print(f'Cached run rejected: {error}. Training a fresh run.',flush=True)
    out.mkdir(exist_ok=False);started=time.perf_counter();rows=[];champions=[];references=[]
    for seed in range(args.seeds):
        pop=np.random.default_rng(710+seed).normal(0,.45,(4,genes));champ=pop[0].copy();stream=np.random.default_rng(1201+seed*2)
        test=np.arange(128)+90000000+seed*1000
        direct=evaluate(np.zeros((len(test),genes)),test,world,policy='direct',pellet_count=count,food_layout=args.food_layout,grid_cells=grid_cells,controller=controller)
        if count==1 and args.food_layout=='nearby' and not direct[:,0].all():raise AssertionError('One-pellet reference failed')
        references.append(float(direct[:,2].mean()))
        for generation in range(args.generations+1):
            if generation:
                ids=np.arange(8)+1000000+generation*10000+seed*100
                result=evaluate(np.repeat(pop,8,axis=0),np.tile(ids,4),world,pellet_count=count,food_layout=args.food_layout,grid_cells=grid_cells,controller=controller).reshape(4,8,4)
                scores=result[:,:,2]+weight*(result[:,:,3]-1)
                champ=pop[scores.mean(axis=1).argmax()].copy();children=[champ.copy()]
                for _ in range(3):
                    child=champ.copy();index=stream.choice(genes,mutation_count,replace=False);child[index]+=stream.normal(0,.35,mutation_count);children.append(child)
                pop=np.array(children)
            if generation%5==0 or generation==args.generations:
                result=evaluate(np.repeat(champ[None],len(test),axis=0),test,world,pellet_count=count,food_layout=args.food_layout,grid_cells=grid_cells,controller=controller)
                rows.append(dict(seed=seed,generation=generation,success=float(result[:,0].mean()),capped_time=float(result[:,1].mean()),food=float(result[:,2].mean()),fraction=float(result[:,2].mean()/count),visited=float(result[:,3].mean()),fitness=float((result[:,2]+weight*(result[:,3]-1)).mean())))
                if controller in ('recurrent','visited-map'):
                    reset=evaluate(np.repeat(champ[None],len(test),axis=0),test,world,pellet_count=count,food_layout=args.food_layout,grid_cells=grid_cells,controller=controller,reset_memory=controller=='recurrent',blank_map=controller=='visited-map')
                    reset_rows.append(dict(seed=seed,generation=generation,success=float(reset[:,0].mean()),capped_time=float(reset[:,1].mean()),food=float(reset[:,2].mean()),fraction=float(reset[:,2].mean()/count),visited=float(reset[:,3].mean()),fitness=float((reset[:,2]+weight*(reset[:,3]-1)).mean())))
                    with (out/'measurements-reset.csv').open('w',newline='') as f:
                        w=csv.DictWriter(f,fieldnames=list(reset_rows[0]));w.writeheader();w.writerows(reset_rows)
                with (out/'measurements.csv').open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
                print(f'{count} pellets | seed {seed+1}/{args.seeds} | generation {generation}/{args.generations} | food {result[:,2].mean():.2f}/{count} | {time.perf_counter()-started:.1f}s',flush=True)
        champions.append(champ.tolist())
    (out/'champions.json').write_text(json.dumps(champions));replay(world,np.array(champions[0]),out,count,args.food_layout,grid_cells,weight,controller)
    (out/'design.json').write_text(json.dumps(dict(pellets=count,food_layout=args.food_layout,exploration_reward=weight,grid_cells=grid_cells,terrain=bool(any(world[0].solid)),generations=args.generations,seeds=args.seeds,population=4,training_starts=8,test_starts=128,genes=genes,controller=controller,mutation_count=mutation_count,mutation_sigma=.35,vision=24,seconds=time.perf_counter()-started,fitness='pellets collected + exploration_reward * (unique centre-occupied tiles - 1); starting tile earns no bonus',notes=['No predator. Same mutation, networks, speed and collection code as one-pellet diagnostic.','Food sets are nested and spaced at least 6 apart. Nearby mode starts food 6 to 20 cells away; scattered mode samples the shared reachable sites across the map.','Stops after all pellets are collected or 10 seconds.','Direct reference steers toward nearest visible food; can stop if all remaining food leaves vision. It is a reference, not a global optimum.']),indent=2))
    (out/'reference.json').write_text(json.dumps(references))
    compatibility['artifact_hashes']={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ('measurements.csv','champions.json','design.json','reference.json')}
    if (out/'measurements-reset.csv').exists():compatibility['artifact_hashes']['measurements-reset.csv']=hashlib.sha256((out/'measurements-reset.csv').read_bytes()).hexdigest()
    (out/'compatibility.json').write_text(json.dumps(compatibility,indent=2))
    registry.mkdir(parents=True,exist_ok=True)
    temp=cache_index.with_suffix('.tmp');temp.write_text(json.dumps(dict(source=str(out.resolve()))));temp.replace(cache_index)
    return rows,references


def landscape_comparison(args):
    """Same scattered layouts in an empty arena and body-accessible terrain."""
    args.output.mkdir(parents=True,exist_ok=False);args.food_layout='scattered';count=args.pellet_counts[0]
    terrain=json.loads(args.terrain_map.read_text());terrain_world=setup(terrain)
    empty=dict(resolution=96,solid=[0]*96**2,preset='Empty arena',world_size=8)
    empty_world=setup(empty)
    # Matching spawn and food sites isolates geometry rather than placing
    # impossible food inside terrain. The shared sites span the arena.
    worlds={'empty':(empty_world[0],terrain_world[1],empty_world[2]),'terrain':terrain_world}
    groups={};references={};summary={}
    for name,world in worlds.items():
        folder=args.output/name;folder.mkdir()
        (folder/'terrain.json').write_text(json.dumps(empty if name=='empty' else terrain))
        print(f'Landscape condition: {name}',flush=True)
        groups[name],references[name]=run_count(args,count,folder/f'{count}-pellets',world)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for name,color in [('empty','#188b87'),('terrain','#ba6954')]:
        rows=groups[name];gs=sorted(set(r['generation'] for r in rows));entry={}
        for ax,key,title in zip(axes,['food','fraction','success'],['Pellets collected','Fraction of available food collected','Encounters collecting all food']):
            values=np.array([[r[key] for r in rows if r['seed']==seed] for seed in range(args.seeds)])
            mean,low,high=seed_mean_interval(values);ax.plot(gs,mean,color=color,lw=2,label=name);ax.fill_between(gs,low,high,color=color,alpha=.18)
            ax.set(title=title,xlabel='Generation',ylabel='Pellets' if key=='food' else 'Fraction',ylim=(0,count*1.05 if key=='food' else 1.03))
            entry[key]=dict(initial=float(mean[0]),final=float(mean[-1]),ci95=[float(low[-1]),float(high[-1])],seed_means=values[:,-1].tolist())
        entry['direct_reference_food']=float(np.mean(references[name]));summary[name]=entry
    axes[0].legend();fig.suptitle(f'Scattered food: empty arena versus complex terrain | {args.seeds} seeds, mean and 95% CI')
    fig.savefig(args.output/'landscape-comparison.png',dpi=170);plt.close(fig)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    sites=terrain_world[1]
    (args.output/'design.json').write_text(json.dumps(dict(pellets=count,generations=args.generations,seeds=args.seeds,terrain_map=str(args.terrain_map),food_layout='Uniform random sampling over shared body-accessible sites in largest connected terrain region; min separation 6',site_bounds=[sites.min(axis=0).tolist(),sites.max(axis=0).tolist()],reachable_sites=len(sites),vision=24,occlusion=False,episode_seconds=STEPS*DT,notes=['Identical starts, pellets, initial weights and mutation random streams across conditions.','Food may start outside vision; organism must explore to discover it.','Food placement has no routing or goal-distance assistance. Direction is still straight-line to nearest visible pellet, including through obstacles.','All sites are reachable by the body; disconnected and too-narrow regions are excluded in BOTH arenas.','Same ten-second horizon; all-food completion may be difficult even with good steering.','No predator, hunger, approach rewards or extra memory. Direct reference is greedy visible-food steering, not pathfinding.']),indent=2));print(json.dumps(summary),flush=True)


def exploration_comparison(args):
    args.output.mkdir(parents=True,exist_ok=False);args.food_layout='scattered';count=args.pellet_counts[0]
    terrain=json.loads(args.terrain_map.read_text());terrain_world=setup(terrain)
    empty=dict(resolution=96,solid=[0]*96**2,preset='Empty arena');empty_world=setup(empty)
    worlds={'empty':(empty_world[0],terrain_world[1],empty_world[2]),'terrain':terrain_world}
    weight=args.exploration_reward if args.exploration_reward>0 else .1
    groups={};summary={}
    for geometry,world in worlds.items():
        folder=args.output/geometry;folder.mkdir();(folder/'terrain.json').write_text(json.dumps(empty if geometry=='empty' else terrain))
        for treatment,w in [('food-only',0.),('exploration',weight)]:
            args.exploration_reward=w;dest=folder/treatment;dest.mkdir()
            print(f'Condition: {geometry}, {treatment}, exploration reward {w:g}',flush=True)
            groups[geometry,treatment],_=run_count(args,count,dest/f'{count}-pellets',world)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    paired={}
    for row,geometry in enumerate(worlds):
        for col,key in enumerate(('food','visited')):
            for treatment,color in [('food-only','#ba6954'),('exploration','#188b87')]:
                records=groups[geometry,treatment];gs=sorted(set(r['generation'] for r in records))
                values=np.array([[r[key] for r in records if r['seed']==seed] for seed in range(args.seeds)])
                mean,low,high=seed_mean_interval(values);ax=axes[row,col]
                ax.plot(gs,mean,color=color,lw=2,label=treatment);ax.fill_between(gs,low,high,color=color,alpha=.18)
                ax.set(title=f'{geometry}: '+('food collection' if key=='food' else 'unique tiles visited'),xlabel='Generation',ylabel='Pellets' if key=='food' else 'Tiles',ylim=(0,count*1.05 if key=='food' else args.grid_cells**2));ax.legend(fontsize=8)
                summary.setdefault(f'{geometry}/{treatment}',{})[key]=dict(initial=float(mean[0]),final=float(mean[-1]),ci95=[float(low[-1]),float(high[-1])],seed_means=values[:,-1].tolist())
        a=np.array(summary[f'{geometry}/exploration']['food']['seed_means']);b=np.array(summary[f'{geometry}/food-only']['food']['seed_means'])
        m,l,h=seed_mean_interval((a-b)[:,None]);paired[geometry]=dict(food_difference=float(m[0]),ci95=[float(l[0]),float(h[0])])
    fig.suptitle(f'Exploration reward: does visiting more tiles improve FOOD collection?\n{args.seeds} matched seeds; means and pointwise 95% bootstrap CIs')
    fig.savefig(args.output/'exploration-comparison.png',dpi=170);plt.close(fig)
    summary['paired_food_differences']=paired;(args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    (args.output/'design.json').write_text(json.dumps(dict(seeds=args.seeds,generations=args.generations,pellets=count,grid_cells=args.grid_cells,grid_cell_width=96/args.grid_cells,bonus=weight,fitness='pellets + bonus * (unique centre-occupied cells - 1)',notes=['Grid is evaluator-only; it is NOT an input to the organism.','Starting tile is marked visited but earns no bonus. Subsequent tiles earn bonus only once per encounter.','Visited means body centre occupied the tile at a simulation timestep, not merely that it was visible.','Repeated visits earn nothing; visited flags reset for every encounter.','Same initial genomes, mutation streams and layouts across treatments; no hunger or predator.','Both conditions use the same shared terrain-accessible sites; no vision occlusion.','Food collection is the primary outcome; exploration reward can improve coverage without improving food collection.']),indent=2));print(json.dumps(summary),flush=True)


def memory_comparison(args):
    args.output.mkdir(parents=True,exist_ok=False);args.food_layout='scattered';count=args.pellet_counts[0]
    terrain=json.loads(args.terrain_map.read_text());terrain_world=setup(terrain)
    empty=dict(resolution=96,solid=[0]*96**2,preset='Empty arena');empty_world=setup(empty)
    worlds={'empty':(empty_world[0],terrain_world[1],empty_world[2]),'terrain':terrain_world}
    groups={};summary={};paired={}
    for geometry,world in worlds.items():
        folder=args.output/geometry;folder.mkdir();(folder/'terrain.json').write_text(json.dumps(empty if geometry=='empty' else terrain))
        for controller in ('feedforward','recurrent'):
            args.controller=controller;dest=folder/controller;dest.mkdir()
            print(f'Condition: {geometry}, {controller} ({GENES if controller=="feedforward" else REC_GENES} genes)',flush=True)
            groups[geometry,controller],_=run_count(args,count,dest/f'{count}-pellets',world)
            if controller=='recurrent':
                with (dest/f'{count}-pellets'/'measurements-reset.csv').open() as f:
                    rows=list(csv.DictReader(f))
                groups[geometry,'reset']=[{k:int(v) if k in ('seed','generation') else float(v) for k,v in r.items()} for r in rows]
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for row,geometry in enumerate(worlds):
        for col,key in enumerate(('food','visited')):
            for controller,label,color in [('feedforward','Feedforward','#ba6954'),('recurrent','Recurrent','#188b87'),('reset','Recurrent: reset each step','#7b6aab')]:
                records=groups[geometry,controller];gs=sorted(set(r['generation'] for r in records))
                values=np.array([[r[key] for r in records if r['seed']==seed] for seed in range(args.seeds)])
                mean,low,high=seed_mean_interval(values);ax=axes[row,col]
                ax.plot(gs,mean,color=color,lw=2,label=label);ax.fill_between(gs,low,high,color=color,alpha=.14)
                ax.set(title=f'{geometry}: '+('pellets collected' if key=='food' else 'unique tiles visited'),xlabel='Generation',ylabel='Pellets' if key=='food' else 'Tiles',ylim=(0,count*1.05 if key=='food' else args.grid_cells**2));ax.legend(fontsize=8)
                summary.setdefault(f'{geometry}/{controller}',{})[key]=dict(initial=float(mean[0]),final=float(mean[-1]),ci95=[float(low[-1]),float(high[-1])],seed_means=values[:,-1].tolist())
        recurrent=np.array(summary[f'{geometry}/recurrent']['food']['seed_means'])
        for control in ('feedforward','reset'):
            other=np.array(summary[f'{geometry}/{control}']['food']['seed_means']);m,l,h=seed_mean_interval((recurrent-other)[:,None])
            paired[f'{geometry}/recurrent-minus-{control}']=dict(food_difference=float(m[0]),ci95=[float(l[0]),float(h[0])])
    fig.suptitle(f'Does controller memory improve foraging? {args.seeds} seeds: mean and 95% CI\nReset curves evaluate the SAME recurrent controllers with hidden state cleared every step')
    fig.savefig(args.output/'memory-comparison.png',dpi=170);plt.close(fig)
    summary['paired_food_differences']=paired;(args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    (args.output/'design.json').write_text(json.dumps(dict(seeds=args.seeds,generations=args.generations,pellets=count,exploration_reward=args.exploration_reward,controllers={'feedforward':dict(hidden=8,genes=GENES),'recurrent':dict(hidden=REC_HIDDEN,genes=REC_GENES,recurrence='h[t] = tanh([obs,1] W_in + h[t-1] W_rec); action=tanh([h[t],1] W_out)')},notes=['Same 18 observations including current own velocity; no explored/unexplored map input.','Recurrent state starts at zero separately for every encounter; never shared across candidates, starts or generations.','Reset-each-step is evaluation only, not a third training treatment; ablation never influences selection.','Feedforward has 170 weights; recurrent has 164, so memory does not gain a larger parameter budget. Architectures differ in hidden width.','Same 4 candidates, 8 new training starts, 128 fixed test starts, 50-generation default, mutation of 12 weights with sigma .35.','Paired seeds share layouts and RNG seeds, but genomes have different shapes; initial controller behaviours are not identical.','Food collection is primary; coverage is diagnostic. The reward is unchanged between treatments.','A reset penalty supports use of temporal state, not necessarily spatial mapping or sophisticated memory.','No predator or occlusion; vision 24, ten-second episodes; shared reachable starts/food sites across landscapes.','Pointwise bootstrap intervals resample seed means, not individual games.']),indent=2));print(json.dumps(summary),flush=True)


def visited_map_comparison(args):
    args.output.mkdir(parents=True,exist_ok=False);args.food_layout='scattered';count=args.pellet_counts[0]
    empty=dict(resolution=96,solid=[0]*96**2,preset='Empty arena')
    world=setup(empty);(args.output/'terrain.json').write_text(json.dumps(empty))
    groups={};summary={};paired={}
    for controller in ('position-only','visited-map'):
        args.controller=controller;folder=args.output/controller;folder.mkdir()
        print(f'Condition: {controller} | {map_gene_count(args.grid_cells)} weights | empty arena',flush=True)
        groups[controller],_=run_count(args,count,folder/f'{count}-pellets',world)
        if controller=='visited-map':
            with (folder/f'{count}-pellets'/'measurements-reset.csv').open() as f:rows=list(csv.DictReader(f))
            groups['map-blanked']=[{k:int(v) if k in ('seed','generation') else float(v) for k,v in r.items()} for r in rows]
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,key,title in zip(axes,('food','visited'),('Food collection','Unique tiles visited')):
        for name,label,color in [('position-only','Position, no visit history','#ba6954'),('visited-map','Position + visited map','#188b87'),('map-blanked','Map controller: history blanked','#7b6aab')]:
            rows=groups[name];gs=sorted(set(r['generation'] for r in rows))
            values=np.array([[r[key] for r in rows if r['seed']==seed] for seed in range(args.seeds)])
            mean,low,high=seed_mean_interval(values);ax.plot(gs,mean,color=color,lw=2,label=label);ax.fill_between(gs,low,high,color=color,alpha=.15)
            ax.set(title=title,xlabel='Generation',ylabel='Pellets' if key=='food' else 'Tiles',ylim=(0,None));ax.legend(fontsize=8)
            summary.setdefault(name,{})[key]=dict(initial=float(mean[0]),final=float(mean[-1]),ci95=[float(low[-1]),float(high[-1])],seed_means=values[:,-1].tolist())
    for control in ('position-only','map-blanked'):
        for key in ('food','visited'):
            difference=np.array(summary['visited-map'][key]['seed_means'])-np.array(summary[control][key]['seed_means'])
            m,l,h=seed_mean_interval(difference[:,None]);paired[f'map-minus-{control}/{key}']=dict(difference=float(m[0]),ci95=[float(l[0]),float(h[0])])
    fig.suptitle(f'Explicit visit memory in an EMPTY arena | {args.seeds} seeds, mean and 95% CI\nBoth trained groups use the SAME reward; exploration bonus = {args.exploration_reward:g}')
    fig.savefig(args.output/'visited-map-comparison.png',dpi=170);plt.close(fig)
    summary['paired_differences']=paired;(args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    (args.output/'design.json').write_text(json.dumps(dict(seeds=args.seeds,generations=args.generations,pellets=count,grid_cells=args.grid_cells,genes=map_gene_count(args.grid_cells),inputs=INPUTS+2+args.grid_cells**2,hidden=8,exploration_reward=args.exploration_reward,notes=['Empty arena only; scattered food over all body-accessible empty-map sites. No predator or terrain.','Both trained networks receive the same 18 local observations and normalised absolute x/y position.','Visited-map treatment additionally receives all grid_cells^2 visit bits. The matched control has those channels fixed to zero.','Map updates from centre occupancy, starts with starting tile marked, resets each encounter. It shows no food or unvisited terrain contents.','Both networks have identical shapes, initial weights, starts and mutation RNG streams; no recurrent state.','Mutation keeps original per-weight rate 12/170 (rounded): 49 weights for the 698-weight default, Gaussian sigma .35.','Map-blanked curves evaluate the same map-trained genomes with history zeroed, while preserving position and local observations; never used for selection.','Reward is identical between treatments; default is food only, with no exploration bonus.','Current-velocity input remains present. Access to absolute position is supplied in BOTH treatments to isolate visit history.','Results compare explicit memory representations and do not directly isolate capacity versus the older 170-weight controller.','Full grid is exposed to the network as requested; evolution still chooses actions, with no hand-coded search or routing rule.']),indent=2));print(json.dumps(summary),flush=True)


def hybrid_experiment(args):
    args.output.mkdir(parents=True,exist_ok=False);args.food_layout='scattered';args.controller='hybrid';count=args.pellet_counts[0]
    terrain=json.loads(args.terrain_map.read_text()) if args.hybrid_arena=='terrain' else dict(resolution=96,solid=[0]*96**2,preset='Empty arena')
    world=setup(terrain);(args.output/'terrain.json').write_text(json.dumps(terrain))
    rows,references=run_count(args,count,args.output/f'{count}-pellets',world)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');summary={}
    for ax,key,title in zip(axes,('food','visited'),('Food collection','Unique tiles visited')):
        gs=sorted(set(r['generation'] for r in rows));values=np.array([[r[key] for r in rows if r['seed']==seed] for seed in range(args.seeds)])
        mean,low,high=seed_mean_interval(values);ax.plot(gs,mean,lw=2,color='#188b87',label='Hybrid: evolved steering/speed');ax.fill_between(gs,low,high,color='#188b87',alpha=.2)
        ax.axhline(mean[0],color='gray',ls=':',label='Unevolved hybrid mean')
        if key=='food':
            m,l,h=seed_mean_interval(np.array(references)[:,None]);ax.axhline(m[0],color='#b39a35',ls='--',label='Same rules: direct scripted steering');ax.axhspan(l[0],h[0],color='#b39a35',alpha=.1)
        ax.set(title=title,xlabel='Generation',ylabel='Pellets' if key=='food' else 'Tiles',ylim=(0,None));ax.legend(fontsize=8)
        summary[key]=dict(initial=float(mean[0]),final=float(mean[-1]),ci95=[float(low[-1]),float(high[-1])],seed_means=values[:,-1].tolist())
        m,l,h=seed_mean_interval((values[:,-1]-values[:,0])[:,None]);summary[key]['learned_gain']=dict(mean=float(m[0]),ci95=[float(l[0]),float(h[0])])
    summary['scripted_reference_food']=float(np.mean(references))
    fig.suptitle(f'Explicit forage/explore instructions: {args.hybrid_arena} arena | {args.seeds} seeds, mean and 95% CI\nGoal selection and heading constraints are programmed; only low-level movement evolves')
    fig.savefig(args.output/'hybrid-comparison.png',dpi=170);plt.close(fig)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    (args.output/'instructions.json').write_text(json.dumps(HYBRID_INSTRUCTIONS,indent=2))
    (args.output/'design.json').write_text(json.dumps(dict(seeds=args.seeds,generations=args.generations,pellets=count,grid_cells=args.grid_cells,genes=GENES,arena=args.hybrid_arena,terrain_map=str(args.terrain_map) if args.hybrid_arena=='terrain' else None,food_layout='scattered over body-accessible sites',fitness='Pellets collected only',programmed='Nearest visible food, otherwise persistent nearest unvisited tile-centre target; least-recently-occupied fallback. Heading cone and minimum throttle enforced.',evolved='170 NN weights selecting heading offset and throttle within constraints',reference='Same goal rules, direct heading, full speed. A competent reference, not a proven optimal search policy.',notes=['No predator; vision 24 with no occlusion. Terrain condition uses the same collision solver and largest reachable region. Visit map and position are used by the goal selector, not exposed as raw NN bits.','NN receives target unit direction, relative distance, forage/explore mode, own velocity and boundary rays.','No unseen-food information enters the goal selector. Terrain exploration goals use a clear representative point per reachable tile; no pathfinding. The 60-degree heading constraint can prevent necessary detours.','Default 10-second episodes, 4 candidates, 8 fresh training starts, 128 held-out starts per seed; mutations unchanged at 12 weights/sigma .35.','Initial hybrid performance already includes programmed goal priorities and progress constraints. Improvement over that initial policy, not all collection, is credited to evolution.','Replays show a purple current target and forage/explore mode alongside the visited grid.']),indent=2));print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('results/pellet-count-sweep'));p.add_argument('--generations',type=int,default=50);p.add_argument('--seeds',type=int,default=10);p.add_argument('--pellet-counts',type=int,nargs='+',default=[1],choices=range(1,13));p.add_argument('--baseline',type=Path);p.add_argument('--food-layout',choices=('nearby','scattered'),default='nearby');p.add_argument('--compare-landscapes',action='store_true');p.add_argument('--exploration-reward',type=float,default=0.);p.add_argument('--grid-cells',type=int,choices=(4,8,12,16),default=8);p.add_argument('--compare-exploration',action='store_true');p.add_argument('--compare-memory',action='store_true');p.add_argument('--controller',choices=('feedforward','recurrent','visited-map','position-only','hybrid'),default='feedforward');p.add_argument('--compare-visited-map',action='store_true');p.add_argument('--hybrid-test',action='store_true');p.add_argument('--hybrid-arena',choices=('empty','terrain'),default='empty');p.add_argument('--reuse-control',type=Path);p.add_argument('--terrain-map',type=Path,default=Path('results/landscape-previews/map-0-seed-42.json'));args=p.parse_args()
    if min(args.generations,args.seeds)<1 or args.exploration_reward<0:p.error('Use positive generations/seeds and nonnegative exploration reward')
    if args.hybrid_test:
        if len(args.pellet_counts)!=1 or args.baseline or args.exploration_reward!=0:p.error('Hybrid test uses one pellet count and food-only reward, with no --baseline')
        hybrid_experiment(args)
        return
    if args.reuse_control and not args.compare_visited_map:p.error('--reuse-control applies to --compare-visited-map')
    if args.compare_visited_map:
        if len(args.pellet_counts)!=1 or args.baseline:p.error('Visited-map comparison requires one pellet count and no --baseline')
        visited_map_comparison(args)
        return
    if args.compare_memory:
        if len(args.pellet_counts)!=1 or args.baseline:p.error('Memory comparison requires one pellet count and no --baseline')
        memory_comparison(args)
        return
    if args.compare_exploration:
        if len(args.pellet_counts)!=1 or args.baseline:p.error('Exploration comparison requires one pellet count and no --baseline')
        exploration_comparison(args)
        return
    if args.compare_landscapes:
        if len(args.pellet_counts)!=1 or args.baseline:p.error('Landscape comparison uses one pellet count and no --baseline')
        landscape_comparison(args)
        return
    if args.baseline and args.food_layout!='nearby':p.error('Nearby baseline cannot be combined with scattered layout')
    args.output.mkdir(parents=True,exist_ok=False);world=setup(dict(resolution=96,solid=[0]*96**2));groups={};refs={}
    if args.baseline:
        design=json.loads((args.baseline/'design.json').read_text())
        if design['generations']!=args.generations or design['seeds']!=args.seeds:p.error('Baseline must match seeds and generations')
        baseline=list(csv.DictReader((args.baseline/'measurements.csv').open()))
        groups[1]=[dict(seed=int(r['seed']),generation=int(r['generation']),food=float(r['success']),fraction=float(r['success']),success=float(r['success'])) for r in baseline]
        refs[1]=[1.]*args.seeds
    for count in dict.fromkeys(args.pellet_counts):
        if count in groups:continue
        groups[count],refs[count]=run_count(args,count,args.output/f'{count}-pellets',world)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained');summary={}
    for color,(count,rows) in zip(plt.rcParams['axes.prop_cycle'].by_key()['color'],sorted(groups.items())):
        gs=sorted(set(r['generation'] for r in rows));entry={}
        for ax,key,title in zip(axes,['food','fraction','success'],['Pellets collected','Fraction of available food collected','Fraction of encounters collecting ALL food']):
            values=np.array([[r[key] for r in rows if r['seed']==seed] for seed in range(args.seeds)])
            mean,low,high=seed_mean_interval(values);ax.plot(gs,mean,color=color,lw=2,label=f'{count} pellets');ax.fill_between(gs,low,high,color=color,alpha=.16)
            ax.set(title=title,xlabel='Generation',ylabel='Pellets' if key=='food' else 'Fraction',ylim=(0,None));entry[key]=dict(initial=float(mean[0]),final=float(mean[-1]),final_ci95=[float(low[-1]),float(high[-1])],final_seed_means=values[:,-1].tolist())
            if key!='food':ax.set_ylim(0,1.03)
        entry['direct_reference_food']=float(np.mean(refs[count]));summary[count]=entry
    axes[0].set_ylim(0,max(groups)*1.05);axes[0].legend();fig.suptitle(f'Adding pellets without predators: means and pointwise 95% bootstrap CIs across {args.seeds} seeds')
    fig.savefig(args.output/'pellet-count-comparison.png',dpi=170);plt.close(fig)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2));(args.output/'design.json').write_text(json.dumps(dict(baseline=str(args.baseline) if args.baseline else None,pellet_counts=sorted(groups),generations=args.generations,seeds=args.seeds,notes=['Reused one-pellet baseline matches starts, network and mutation settings.','More pellets increases food density and changes nearest-food targets; collected count alone is not a pure measure of steering improvement.']),indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
