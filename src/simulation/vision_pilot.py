"""Matched vision-range pilot on one exported terrain; evolve NN weights only."""
from src.analysis.replay_fonts import replay_font
import argparse, csv, json, time
from pathlib import Path
from collections import deque
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.terrain.grid_collision import GridCollider

CONDITIONS = {'equal':(24.,24.), 'prey-long':(24.,48.), 'predator-long':(48.,24.)}
INPUTS,HIDDEN=18,8
PELLET_COUNT, FOOD_REWARD, FOOD_RADIUS = 12, .25, 3.4
GENES=(INPUTS+1)*HIDDEN+(HIDDEN+1)*2
DT,STEPS,SPEED=.1,100,18.

def food_observation(position, pellets, available, vision, encoding="original"):
    delta=pellets-position[:,None,:]
    distance=np.linalg.norm(delta,axis=-1)
    visible=available & (distance<=vision)
    index=np.where(visible,distance,np.inf).argmin(axis=1)
    found=visible.any(axis=1)
    target=delta[np.arange(len(position)),index]
    if encoding == 'unit':
        direction=target/np.maximum(np.linalg.norm(target,axis=1,keepdims=True),1e-12)
    elif encoding == 'original':
        direction=target/96
    else:
        raise ValueError(f'Unknown food encoding: {encoding}')
    return np.c_[direction*found[:,None],found]


def food_layout(ids,sites,positions):
    layouts=[]
    for seed,position in zip(ids,positions):
        rng=np.random.default_rng(int(seed)+900000)
        candidates=sites[np.linalg.norm(sites-position[1],axis=1)>FOOD_RADIUS+2]
        chosen=[]
        for index in rng.permutation(len(candidates)):
            point=candidates[index]
            if all(np.linalg.norm(point-other)>=6 for other in chosen):chosen.append(point)
            if len(chosen)==PELLET_COUNT:break
        if len(chosen)!=PELLET_COUNT:raise ValueError('Not enough space for food')
        layouts.append(chosen)
    return np.array(layouts)


def collect_food(previous,current,pellets,available,active):
    motion=current-previous
    fraction=np.clip(np.sum((pellets-previous[:,None])*motion[:,None],axis=-1)/np.maximum(np.sum(motion*motion,axis=-1)[:,None],1e-12),0,1)
    distance=np.linalg.norm(pellets-(previous[:,None]+fraction[:,:,None]*motion[:,None]),axis=-1)
    eaten=available & active[:,None] & (distance<=FOOD_RADIUS)
    available[eaten]=False
    return eaten.sum(axis=1)


def observe(pos,velocity,ranges,rays,pellets=None,available=None,food_encoding="original"):
    obs=[]; visible=[]
    for side in range(2):
        delta=pos[:,1-side]-pos[:,side]
        seen=np.linalg.norm(delta,axis=1)<=ranges[side]
        cells=np.clip(pos[:,side].astype(int),0,95)
        food=np.zeros((len(pos),3))
        if side==1 and pellets is not None:food=food_observation(pos[:,side],pellets,available,ranges[side],food_encoding)
        obs.append(np.c_[delta/96*seen[:,None],(velocity[:,1-side]-velocity[:,side])/36*seen[:,None],seen,velocity[:,side]/SPEED,rays[cells[:,1],cells[:,0]],food])
        visible.append(seen)
    return obs,np.stack(visible,axis=1)

def network(g,obs):
    cut=(INPUTS+1)*HIDDEN
    w=g[:,:cut].reshape(-1,INPUTS+1,HIDDEN)
    v=g[:,cut:].reshape(-1,HIDDEN+1,2)
    h=np.tanh(np.einsum('ni,nij->nj',np.c_[obs,np.ones(len(g))],w))
    return np.tanh(np.einsum('ni,nij->nj',np.c_[h,np.ones(len(g))],v))

def setup(data):
    collider=GridCollider(data,radius=2.4)
    clear={(x,y) for y in range(96) for x in range(96) if collider.clear((x+.5,y+.5))}
    groups=[]
    while clear:
        root=min(clear); clear.remove(root); group=[root]; q=deque([root])
        while q:
            x,y=q.popleft()
            for cell in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                if cell in clear:
                    clear.remove(cell);group.append(cell);q.append(cell)
        groups.append(group)
    sites=np.array(sorted(max(groups,key=len)),float)+.5
    yy,xx=np.mgrid[:96,:96]; origins=np.stack((xx+.5,yy+.5),axis=-1)
    solid=np.array(data['solid']).reshape(96,96); rays=np.ones((96,96,8))
    for i,angle in enumerate(np.arange(8)*np.pi/4):
        hit=np.zeros((96,96),bool)
        for distance in range(1,25):
            p=np.floor(origins+distance*np.array([np.cos(angle),np.sin(angle)])).astype(int)
            outside=((p<0)|(p>=96)).any(axis=-1)
            blocked=outside|solid[np.clip(p[:,:,1],0,95),np.clip(p[:,:,0],0,95)].astype(bool)
            rays[:,:,i]=np.where(blocked&~hit,distance/24,rays[:,:,i]);hit|=blocked
    return collider,sites,rays

def starts(ids,sites):
    result=[]
    for seed in ids:
        rng=np.random.default_rng(int(seed));a=sites[rng.integers(len(sites))]
        d=np.linalg.norm(sites-a,axis=1); choices=sites[(d>=18)&(d<=40)]
        result.append([a,choices[rng.integers(len(choices))]])
    return np.array(result)

def training_ids(seed, generation, mode='resampled', count=2):
    """Share encounters across candidates; reserve 5000+ for evaluation."""
    if mode == 'fixed' or generation <= 1:
        return np.arange(count) + seed * 100
    return np.arange(count) + 1000000 + (generation - 2) * 1000 + seed * 10


def episodes(a,b,ids,ranges,world,trace=None,capture=True,mask_food=False,food_encoding="original",mask_prey_opponent=False):
    collider,sites,rays=world; pos=starts(ids,sites); velocity=np.zeros_like(pos)
    pellets=food_layout(ids,sites,pos);available=np.ones((len(a),PELLET_COUNT),bool);food=np.zeros(len(a))
    n=len(a); active=np.ones(n,bool);caught=np.zeros(n,bool);end=np.full(n,STEPS*DT)
    nearest=np.linalg.norm(pos[:,0]-pos[:,1],axis=1); sightings=np.zeros((n,2))
    for tick in range(STEPS):
        obs,seen=observe(pos,velocity,ranges,rays,pellets,available,food_encoding)
        if not capture:
            for inputs in obs:inputs[:,:5]=0
            seen[:]=False
        if mask_prey_opponent:
            obs[1][:,:5]=0
            seen[:,1]=False
        if mask_food:obs[1][:,-3:]=0
        sightings+=seen*active[:,None]
        if trace is not None: trace.append({'positions':pos[0].tolist(),'visible':seen[0].tolist(),'caught':bool(caught[0]),'pellets':pellets[0,available[0]].tolist(),'food':int(food[0])})
        if not active.any():continue
        desired=np.stack([network(a,obs[0]),network(b,obs[1])],axis=1)
        desired*=SPEED/np.maximum(1,np.linalg.norm(desired,axis=-1,keepdims=True))
        if not capture:desired[:,0]=0
        old=pos.copy()
        for row in np.flatnonzero(active):
            for side in (range(2) if capture else (1,)):pos[row,side]=collider.move(pos[row,side],desired[row,side]*DT)[0]
        velocity=(pos-old)/DT
        rel=old[:,0]-old[:,1];travel=pos[:,0]-pos[:,1]-rel
        t=np.clip(-np.sum(rel*travel,axis=1)/np.maximum(np.sum(travel*travel,axis=1),1e-12),0,1)
        distance=np.linalg.norm(rel+t[:,None]*travel,axis=1);nearest=np.minimum(nearest,distance)
        tag=active & capture & (distance<=collider.radius*2);caught|=tag;end[tag]=(tick+1)*DT;active[tag]=False
        food+=collect_food(old[:,1],pos[:,1],pellets,available,active)
    fraction=end/(STEPS*DT)
    return np.c_[caught*(2-fraction)+.2*np.exp(-nearest/12),fraction-caught+FOOD_REWARD*food,caught,end,sightings/np.maximum(end[:,None]/DT,1),food]

def render(data,examples,out,index):
    tile=320; top=64; font=replay_font(15)
    mask=np.array(data['solid']).reshape(96,96)
    bg=Image.fromarray(np.where(mask[:,:,None],np.array([108,119,116]),np.array([232,228,206])).astype('uint8')).resize((tile,tile),Image.Resampling.NEAREST)
    frames=[]
    for tick in range(0,STEPS,2):
        im=Image.new('RGB',(tile*3,top+tile+48),'#172321');d=ImageDraw.Draw(im)
        d.text((10,7),'Vision pilot | coral: predator | teal: prey | bright body = opponent visible',font=font,fill='white')
        for panel,(name,trace) in enumerate(examples.items()):
            x=panel*tile;im.paste(bg,(x,top));d=ImageDraw.Draw(im,'RGBA')
            d.text((x+8,36),name,font=font,fill='white');f=trace[tick]
            for fx,fy in f['pellets']:
                px=x+fx*tile/96;py=top+fy*tile/96
                d.ellipse((px-3,py-3,px+3,py+3),fill='#e3b72f',outline='#66551a')
            for side,color in enumerate(((225,117,94),(35,155,157))):
                px,py=f['positions'][side];px=x+px*tile/96;py=top+py*tile/96;r=CONDITIONS[name][side]*tile/96
                # Clip range overlays to this map panel.
                layer=Image.new('RGBA',(tile,tile));ld=ImageDraw.Draw(layer)
                ld.ellipse((px-x-r,py-top-r,px-x+r,py-top+r),fill=(*color,18),outline=(*color,160),width=2)
                im.paste(Image.alpha_composite(im.crop((x,top,x+tile,top+tile)).convert('RGBA'),layer).convert('RGB'),(x,top))
                d=ImageDraw.Draw(im,'RGBA');body=2.4*tile/96
                d.ellipse((px-body,py-body,px+body,py+body),fill=(*color,255),outline='white' if f['visible'][side] else '#263330',width=3)
            d.text((x+8,top+tile+8),f'{tick*DT:.1f}s | '+('captured' if f['caught'] else 'active')+f" | food {f['food']}",font=font,fill='white')
        frames.append(im)
    frames[0].save(out/f'vision-replay-{index}.gif',save_all=True,append_images=frames[1:],duration=200,loop=0)

def population_test(args):
    """Isolate population size in foraging; no capture or opponent inputs."""
    out=args.output;out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    data=json.loads(Path('results/landscape-previews/map-0-seed-42.json').read_text())
    world=setup(data);(out/'terrain.json').write_text(json.dumps(data))
    rows=[];champions={};checks=[]
    def evaluate(genomes,ids,mask_food=False):
        prey=np.repeat(genomes,len(ids),axis=0)
        return episodes(np.zeros_like(prey),prey,np.tile(ids,len(genomes)),(24,24),world,capture=False,mask_food=mask_food).reshape(len(genomes),len(ids),7)
    for size in (4,16):
        for seed in range(args.seeds):
            initial=np.random.default_rng(710+seed).normal(0,.45,(size,GENES))
            pop=initial.copy();champ=initial[0].copy();rng=np.random.default_rng(1201+2*seed)
            test_ids=np.arange(24)+5000+seed*100
            for generation in range(args.generations+1):
                if generation:
                    ids=training_ids(seed,generation,args.training_mode)
                    scores=evaluate(pop,ids)[:,:,1].mean(axis=1)
                    champ=pop[scores.argmax()].copy();children=[champ.copy()]
                    for _ in range(size-1):
                        child=champ.copy();index=rng.choice(GENES,12,replace=False)
                        child[index]+=rng.normal(0,.35,12);children.append(child)
                    pop=np.array(children)
                if generation % args.eval_every==0 or generation==args.generations:
                    result=evaluate(champ[None],test_ids)[0]
                    for trial,row in enumerate(result):
                        rows.append(dict(population=size,seed=seed,generation=generation,start=int(test_ids[trial]),food=float(row[6])))
                    with (out/'measurements.csv').open('w',newline='') as f:
                        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                    print(f'Foraging | population {size} | seed {seed+1}/{args.seeds} | generation {generation}/{args.generations} | held-out food {result[:,6].mean():.3f} | {time.perf_counter()-started:.1f}s',flush=True)
            champions[f'{size}/{seed}']=champ.tolist()
            unseen=evaluate(champ[None],test_ids,mask_food=True)[0,:,6].mean()
            checks.append(dict(population=size,seed=seed,food_sensor_off=float(unseen)))
    (out/'champions.json').write_text(json.dumps(champions))
    summary={}
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for size,color in ((4,'#b46a50'),(16,'#188b87')):
        gs=sorted(set(row['generation'] for row in rows));values=[]
        for seed in range(args.seeds):
            ys=[np.mean([r['food'] for r in rows if r['population']==size and r['seed']==seed and r['generation']==g]) for g in gs]
            values.append(ys);axes[0].plot(gs,ys,color=color,alpha=.25)
        values=np.array(values);axes[0].plot(gs,values.mean(axis=0),color=color,lw=2.5,label=f'{size} controllers')
        final=values[:,-1];off=[r['food_sensor_off'] for r in checks if r['population']==size]
        axes[1].scatter(np.full(len(final),size),final,color=color,s=45)
        axes[1].plot([size-.9,size+.9],[final.mean()]*2,color=color,lw=3)
        summary[size]=dict(initial_food=float(values[:,0].mean()),final_food=float(final.mean()),final_seed_means=final.tolist(),food_sensor_off=float(np.mean(off)))
    axes[0].set(title='Food collection across generations',xlabel='Generation',ylabel='Pellets per encounter',ylim=(0,None));axes[0].legend()
    axes[1].set(title='Final collection: dots = evolutionary seeds',xlabel='Population size',ylabel='Pellets per encounter',xticks=[4,16],xlim=(1,19),ylim=(0,None))
    fig.suptitle('Foraging only: 4 versus 16 candidate controllers (capture disabled)')
    fig.savefig(out/'population-comparison.png',dpi=170);plt.close(fig)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    design=dict(generations=args.generations,seeds=args.seeds,populations=[4,16],training_mode=args.training_mode,training_starts=2,test_starts=24,capture=False,opponent_inputs=False,vision=24,food_reward=FOOD_REWARD,food_sensing='Original nearest-visible displacement divided by 96 plus availability flag',approach_reward=0,mutation='12 weights changed per child, sigma .35; one elite parent',notes=['Same initial first controller, map, training schedule and evaluation starts across population sizes.','Equal generations; population 16 uses four times the training evaluations.','No predator pairings needed for this isolated foraging diagnostic.'],seconds=time.perf_counter()-started)
    (out/'design.json').write_text(json.dumps(design,indent=2));print(json.dumps(summary),flush=True)


def factor_test(args):
    """Paired 2x2 test of food representation and terrain, with no capture."""
    out=args.output;out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    terrain=json.loads(Path('results/landscape-previews/map-0-seed-42.json').read_text())
    empty=dict(terrain,solid=[0]*96**2,preset='Empty arena')
    terrain_world=setup(terrain);empty_world=setup(empty)
    # Use the terrain-valid spawn/food sites in BOTH arenas. Every seed sees
    # exactly the same starts and pellets; only obstacle geometry differs.
    worlds={'terrain':terrain_world,'empty':(empty_world[0],terrain_world[1],empty_world[2])}
    for name,data in [('terrain',terrain),('empty',empty)]:
        (out/f'{name}.json').write_text(json.dumps(data))
    rows=[];champions={};summary={};sensor_checks=[];size=16
    def evaluate(genomes,ids,world,encoding,mask=False):
        prey=np.repeat(genomes,len(ids),axis=0)
        return episodes(np.zeros_like(prey),prey,np.tile(ids,len(genomes)),(24,24),world,capture=False,mask_food=mask,food_encoding=encoding).reshape(len(genomes),len(ids),7)
    for geometry,world in worlds.items():
        for encoding in ('original','unit'):
            label=f'{geometry}/{encoding}'
            for seed in range(args.seeds):
                pop=np.random.default_rng(710+seed).normal(0,.45,(size,GENES))
                champ=pop[0].copy();rng=np.random.default_rng(1201+seed*2)
                test_ids=np.arange(24)+5000+seed*100
                for generation in range(args.generations+1):
                    if generation:
                        ids=training_ids(seed,generation,args.training_mode)
                        scores=evaluate(pop,ids,world,encoding)[:,:,1].mean(axis=1)
                        champ=pop[scores.argmax()].copy();children=[champ.copy()]
                        for _ in range(size-1):
                            child=champ.copy();index=rng.choice(GENES,12,replace=False)
                            child[index]+=rng.normal(0,.35,12);children.append(child)
                        pop=np.array(children)
                    if generation % args.eval_every==0 or generation==args.generations:
                        result=evaluate(champ[None],test_ids,world,encoding)[0]
                        for trial,row in enumerate(result):rows.append(dict(geometry=geometry,encoding=encoding,seed=seed,generation=generation,start=int(test_ids[trial]),food=float(row[6])))
                        with (out/'measurements.csv').open('w',newline='') as f:
                            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
                        print(f'{label} | seed {seed+1}/{args.seeds} | generation {generation}/{args.generations} | food {result[:,6].mean():.3f} | {time.perf_counter()-started:.1f}s',flush=True)
                champions[f'{label}/{seed}']=champ.tolist()
                off=evaluate(champ[None],test_ids,world,encoding,True)[0,:,6].mean()
                sensor_checks.append(dict(geometry=geometry,encoding=encoding,seed=seed,food_sensor_off=float(off)))
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,geometry in zip(axes,('empty','terrain')):
        for encoding,color in [('original','#b46a50'),('unit','#188b87')]:
            gs=sorted(set(r['generation'] for r in rows));values=[]
            for seed in range(args.seeds):
                ys=[np.mean([r['food'] for r in rows if r['geometry']==geometry and r['encoding']==encoding and r['seed']==seed and r['generation']==g]) for g in gs]
                values.append(ys);ax.plot(gs,ys,color=color,alpha=.22)
            values=np.array(values);ax.plot(gs,values.mean(axis=0),color=color,lw=2.5,label=encoding)
            off=[r['food_sensor_off'] for r in sensor_checks if r['geometry']==geometry and r['encoding']==encoding]
            summary[f'{geometry}/{encoding}']=dict(initial_food=float(values[:,0].mean()),final_food=float(values[:,-1].mean()),final_seed_means=values[:,-1].tolist(),food_sensor_off=float(np.mean(off)))
        ax.set(title='Empty arena' if geometry=='empty' else 'Original terrain',xlabel='Generation',ylabel='Pellets per encounter',ylim=(0,None));ax.legend(title='Food direction input')
    ceiling=max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:ax.set_ylim(0,ceiling)
    fig.suptitle('Food-input and obstacle diagnostic: 16 controllers, no capture\nThick lines = seed means; faint lines = individual evolutionary seeds')
    fig.savefig(out/'factor-comparison.png',dpi=170);plt.close(fig)
    (out/'champions.json').write_text(json.dumps(champions));(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'sensor-checks.json').write_text(json.dumps(sensor_checks,indent=2))
    design=dict(generations=args.generations,seeds=args.seeds,population=size,training_mode=args.training_mode,training_starts=2,test_starts=24,capture=False,vision=24,food_reward=FOOD_REWARD,approach_reward=0,encodings={'original':'displacement / 96 plus visible flag','unit':'unit direction plus visible flag; direction magnitude no longer encodes distance'},notes=['Same 170-weight network and mutation settings in all four conditions.','Spawn positions and food layouts match exactly between arenas, sampled from terrain-valid sites.','Empty arena retains outer boundaries but removes interior terrain and its sensor readings.','Evaluation remains fixed; two training encounters per generation still create selection noise.','Final endpoint is reported without choosing a best evaluation checkpoint.'],seconds=time.perf_counter()-started)
    (out/'design.json').write_text(json.dumps(design,indent=2));print(json.dumps(summary),flush=True)


def seed_mean_interval(values, draws=10000):
    """Pointwise percentile bootstrap over complete independent seed trajectories."""
    values=np.asarray(values,dtype=float)
    if values.ndim!=2 or len(values)<1 or not np.isfinite(values).all():
        raise ValueError('Expected finite seed-by-checkpoint measurements')
    indices=np.random.default_rng(290619).integers(0,len(values),size=(draws,len(values)))
    boot=values[indices].mean(axis=1)
    low,high=np.percentile(boot,[2.5,97.5],axis=0)
    return values.mean(axis=0),low,high


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('results/vision-pilot'));p.add_argument('--generations',type=int,default=12);p.add_argument('--seeds',type=int,default=3);p.add_argument('--eval-every',type=int,default=10);p.add_argument('--training-mode',choices=('fixed','resampled'),default='resampled');p.add_argument('--foraging-population-test',action='store_true');p.add_argument('--foraging-factor-test',action='store_true');p.add_argument('--arena',choices=('terrain','empty'),default='terrain');p.add_argument('--food-encoding',choices=('original','unit'),default='original');p.add_argument('--training-starts',type=int,default=2,choices=range(1,9));p.add_argument('--conditions',nargs='+',choices=tuple(CONDITIONS),default=list(CONDITIONS));p.add_argument('--opponent-mode',choices=('coevolution','fixed'),default='coevolution');p.add_argument('--seed-offset',type=int,default=0);args=p.parse_args()
    if min(args.generations,args.seeds,args.eval_every)<1:p.error('Generations, seeds and evaluation interval must be positive')
    if args.foraging_factor_test:
        factor_test(args)
        return
    if args.foraging_population_test:
        population_test(args)
        return
    args.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    if args.arena == 'empty':
        data=dict(resolution=96,solid=[0]*96**2,preset='Empty arena',world_size=8.0)
    else:
        data=json.loads(Path('results/landscape-previews/map-0-seed-42.json').read_text())
    world=setup(data)
    (args.output/'terrain.json').write_text(json.dumps(data))
    # Opponent-specific inputs must remain identical when an invisible opponent moves.
    pos=np.array([[[10.,10.],[80.,80.]]]);vel=np.zeros_like(pos)
    one,_=observe(pos,vel,(24,24),world[2]);pos[0,1]=[85,85];vel[0,1]=[5,3]
    two,_=observe(pos,vel,(24,24),world[2]);assert np.array_equal(one[0],two[0]),'Opponent information leak'
    conditions={name:CONDITIONS[name] for name in dict.fromkeys(args.conditions)}
    rows=[]; champions={};history={};replays=[{} for _ in range(2)]
    for name,ranges in conditions.items():
        for seed in range(args.seeds):
            effective_seed=seed+args.seed_offset
            rng=np.random.default_rng(710+effective_seed);initial=rng.normal(0,.45,(4,GENES));pops=[initial.copy(),initial.copy()]
            streams=[np.random.default_rng(1200+effective_seed*2+s) for s in range(2)]
            champ=[initial[0].copy(),initial[0].copy()]
            for generation in range(args.generations+1):
                if generation:
                    encounter_ids=training_ids(effective_seed,generation,args.training_mode,args.training_starts)
                    if args.opponent_mode == 'fixed':
                        # Evolve each role against its frozen generation-zero
                        # opponent. Identical deterministic repeats are omitted.
                        count=4*args.training_starts
                        frozen=np.repeat(initial[0][None],count,axis=0)
                        evolving=[np.repeat(pop,args.training_starts,axis=0) for pop in pops]
                        predator=episodes(evolving[0],frozen,np.tile(encounter_ids,4),ranges,world,food_encoding=args.food_encoding).reshape(4,args.training_starts,7)
                        prey=episodes(frozen,evolving[1],np.tile(encounter_ids,4),ranges,world,food_encoding=args.food_encoding).reshape(4,args.training_starts,7)
                        scores=[predator[:,:,0].mean(axis=1),prey[:,:,1].mean(axis=1)]
                    else:
                        a=np.repeat(pops[0],4*args.training_starts,axis=0);b=np.tile(np.repeat(pops[1],args.training_starts,axis=0),(4,1))
                        v=episodes(a,b,np.tile(encounter_ids,16),ranges,world,food_encoding=args.food_encoding).reshape(4,4,args.training_starts,7)
                        scores=[v[:,:,:,0].mean(axis=(1,2)),v[:,:,:,1].mean(axis=(0,2))]
                    champ=[pop[s.argmax()].copy() for pop,s in zip(pops,scores)]
                    pops=[]
                    for g,r in zip(champ,streams):
                        children=[g.copy()]
                        for _ in range(3):
                            child=g.copy();idx=r.choice(GENES,12,replace=False);child[idx]+=r.normal(0,.35,12);children.append(child)
                        pops.append(np.array(children))
                if generation % args.eval_every == 0 or generation == args.generations:
                    history[f'{name}/{seed}/{generation}']=[g.tolist() for g in champ]
                    (args.output/'history.json').write_text(json.dumps(history))
                    ids=np.arange(12)+5000+effective_seed*100
                    v=episodes(np.repeat(champ[0][None],12,axis=0),np.repeat(champ[1][None],12,axis=0),ids,ranges,world,food_encoding=args.food_encoding)
                    for trial,row in enumerate(v):rows.append(dict(condition=name,seed=seed,generation=generation,start=int(ids[trial]),capture=float(row[2]),duration=float(row[3]),predator_visibility=float(row[4]),prey_visibility=float(row[5]),food=float(row[6])))
                    with (args.output/'measurements.csv').open('w',newline='') as f:
                        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
                    (args.output/'checkpoint.json').write_text(json.dumps(dict(condition=name,seed=seed,generation=generation,populations=[pop.tolist() for pop in pops],champions=[g.tolist() for g in champ],rng_states=[r.bit_generator.state for r in streams])))
                print(f'{name} | seed {seed+1}/{args.seeds} | generation {generation}/{args.generations} | {time.perf_counter()-started:.1f}s',flush=True)
            champions[f'{name}/{seed}']=[g.tolist() for g in champ]
            if seed==0:
                for i in range(2):
                    trace=[];episodes(champ[0][None],champ[1][None],[5000+args.seed_offset*100+i],ranges,world,trace,food_encoding=args.food_encoding);replays[i][name]=trace
    with (args.output/'measurements.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (args.output/'champions.json').write_text(json.dumps(champions))
    for i,replay in enumerate(replays):
        render(data,replay,args.output,i);(args.output/f'replay-{i}.json').write_text(json.dumps(replay))
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for name in conditions:
        for ax,key in zip(axes,['capture','duration','food']):
            values=[]
            for seed in range(args.seeds):
                subset=[r for r in rows if r['condition']==name and r['seed']==seed];gs=sorted(set(r['generation'] for r in subset));ys=[np.mean([r[key] for r in subset if r['generation']==g]) for g in gs];values.append(ys)
            mean,low,high=seed_mean_interval(values)
            color=list(plt.rcParams['axes.prop_cycle'].by_key()['color'])[list(CONDITIONS).index(name)]
            ax.plot(gs,mean,lw=2,label=name,color=color);ax.fill_between(gs,low,high,color=color,alpha=.18)
    axes[0].set(title='Capture fraction: mean and 95% CI',xlabel='Generation',ylabel='Fraction',ylim=(-.03,1.03));axes[1].set(title='Episode duration (10s = survived)',xlabel='Generation',ylabel='Seconds');axes[2].set(title='Pellets collected by prey',xlabel='Generation',ylabel='Pellets',ylim=(0,PELLET_COUNT));axes[0].legend();fig.savefig(args.output/'outcomes.png',dpi=160);plt.close(fig)
    design=dict(conditions=conditions,generations=args.generations,seeds=args.seeds,seed_offset=args.seed_offset,evaluation_interval=args.eval_every,population=4,genes=GENES,seconds=time.perf_counter()-started,terrain=data['preset'],arena=args.arena,capture=True,opponent_mode=args.opponent_mode,food_encoding=args.food_encoding,speed=SPEED,body_radius=2.4,dt=DT,steps=STEPS,occlusion=False,food=True,pellet_count=PELLET_COUNT,food_reward=FOOD_REWARD,food_collection_radius=FOOD_RADIUS,training_starts=args.training_starts,training_mode=args.training_mode,test_starts=12,notes=['Units are map cells: 12 cells per world unit.','Equal speed; direct bounded velocity outputs; no heading or acceleration.','Both opponent displacement and velocity masked outside range; explicit visibility flag.','Eight shared local terrain rays, range 24 cells; terrain sensing does not reveal opponent.','Predator reward: capture speed plus weak nearest-distance shaping; prey: survival fraction minus capture plus .25 per pellet.','Food: 12 finite pellets, deterministic per start; nearest visible pellet direction and availability flag sensed only by prey. No respawn or hunger.',
                     'Training encounters are shared by every candidate pairing; resampled mode changes starts and food layouts each generation after the first. Evaluation IDs remain fixed.',
                     'Elitist mutation-only evolution of NN weights; no crossover.','Replays: seed zero, first two held-out starts; frozen after capture.','Small feasibility pilot, not a conclusive comparison.'])
    (args.output/'design.json').write_text(json.dumps(design,indent=2));print(json.dumps(design),flush=True)

if __name__=='__main__':main()
