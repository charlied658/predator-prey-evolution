"""Interpretable decision trees, fixed navigation, and first-life hunger episodes."""
from copy import deepcopy
import numpy as np
from src.simulation.vision_pilot import DT,FOOD_RADIUS
from src.simulation.single_pellet import hybrid_exploration_sites
from src.simulation.continuous_pursuit import spawn_site,spawn_food
from src.simulation.prey_interactions import pickup_times
from src.terrain.terrain_navigation import navigation
from src.training.hybrid_pursuit import capture_fraction,interpolate_path,flee_waypoint


FEATURES={
    'hunger':(0.,1.),'food_count':(0.,16.),'predator_visible':(0.,1.),
    'food_distance':(0.,96.),'predator_distance':(0.,96.),
    'food_route_distance':(0.,192.),'predator_route_distance':(0.,192.),
    'x':(0.,1.),'y':(0.,1.),'vx':(-1.,1.),'vy':(-1.,1.),
    'explored_fraction':(0.,1.),'target_age':(0.,30.),'stuck_time':(0.,30.)}
PREDATOR_FEATURES={
    'prey_visible':(0.,1.),'prey_distance':(0.,96.),'prey_route_distance':(0.,192.),
    'prey_dx':(-1.,1.),'prey_dy':(-1.,1.),'prey_vx':(-1.,1.),'prey_vy':(-1.,1.),
    'x':(0.,1.),'y':(0.,1.),'vx':(-1.,1.),'vy':(-1.,1.),
    'explored_fraction':(0.,1.),'target_age':(0.,30.),'target_distance':(0.,136.),'stuck_time':(0.,30.)}
DEFAULTS=dict(seconds=30.,initial_hunger=10.,maximum_hunger=14.,metabolism=1.,pellet_energy=6.,
              prey_speed=18.,predator_speed=14.4,prey_vision=36.,predator_vision=24.,
              pellets=16,decision_seconds=.5,memory_seconds=20.,predator=True)


def action(kind,**options):return dict(action=kind,**options)


def random_action(rng,role='prey'):
    kind=str(rng.choice(['approach','away','location','stay','continue'] if role=='predator' else ['approach','away','location','stay']))
    if kind in ('approach','away'):
        if role=='predator':return action(kind,object='prey',rank=0,order=str(rng.choice(['distance','route_distance'])))
        return action(kind,object=str(rng.choice(['food','predator'])),rank=int(rng.integers(3)),order=str(rng.choice(['distance','route_distance'])))
    if kind=='location':return action(kind,selector=str(rng.choice(['unvisited','oldest','random'])))
    return action(kind)


def random_tree(rng,depth=3,role='prey'):
    if depth==0 or rng.random()<.3:return random_action(rng,role)
    features=PREDATOR_FEATURES if role=='predator' else FEATURES
    feature=str(rng.choice(list(features)));lo,hi=features[feature]
    threshold=.5 if feature.endswith('_visible') else float(rng.uniform(lo,hi))
    return dict(feature=feature,threshold=threshold,yes=random_tree(rng,depth-1,role),no=random_tree(rng,depth-1,role))


def decide(tree,features):
    while 'action' not in tree:tree=tree['yes'] if features[tree['feature']]<tree['threshold'] else tree['no']
    return tree


def mutate(tree,rng,max_depth=3,role='prey'):
    features=PREDATOR_FEATURES if role=='predator' else FEATURES
    child=deepcopy(tree);paths=[]
    def visit(node,path):
        paths.append(path)
        if 'action' not in node:
            visit(node['yes'],path+['yes']);visit(node['no'],path+['no'])
    visit(child,[]);path=paths[int(rng.integers(len(paths)))];node=child
    for step in path:node=node[step]
    if 'action' in node or rng.random()<.35:
        replacement=random_tree(rng,max_depth-len(path),role)
        if not path:return replacement
        parent=child
        for step in path[:-1]:parent=parent[step]
        parent[path[-1]]=replacement
    elif rng.random()<.5:
        lo,hi=features[node['feature']]
        node['threshold']=float(np.clip(node['threshold']+rng.normal(0,.15*(hi-lo)),lo,hi))
    elif rng.random()<.7:
        node['feature']=str(rng.choice(list(features)));lo,hi=features[node['feature']]
        node['threshold']=float(rng.uniform(lo,hi))
    else:node['yes'],node['no']=node['no'],node['yes']
    return child


def tree_text(tree,indent=''):
    if 'action' in tree:return indent+json_action(tree)
    return (f'{indent}IF {tree["feature"]} < {tree["threshold"]:.3f}:\n'+tree_text(tree['yes'],indent+'  ')+
            f'\n{indent}ELSE:\n'+tree_text(tree['no'],indent+'  '))


def json_action(a):return ' '.join([a['action']]+[f'{k}={v}' for k,v in a.items() if k!='action'])


BASELINE=dict(feature='predator_distance',threshold=27.,yes=action('away',object='predator',rank=0,order='distance'),
              no=dict(feature='food_count',threshold=.5,yes=action('location',selector='unvisited'),
                      no=action('approach',object='food',rank=0,order='distance')))
PREDATOR_BASELINE=dict(feature='prey_visible',threshold=.5,
                      yes=dict(feature='target_distance',threshold=2.4,yes=action('location',selector='unvisited'),no=action('continue')),
                      no=action('approach',object='prey',rank=0,order='distance'))


class VisitMemory:
    def __init__(self,world):
        self.centres,self.eligible=hybrid_exploration_sites(world,8)
        self.last=np.full(64,-np.inf)

    def mark(self,position,seconds):
        x,y=np.clip((np.asarray(position)*8/96).astype(int),0,7);self.last[y*8+x]=seconds

    def location(self,position,seconds,selector,rng,memory_seconds):
        ids=np.flatnonzero(self.eligible)
        if selector=='random':return self.centres[rng.choice(ids)].copy()
        if selector=='oldest':ids=ids[self.last[ids]==self.last[ids].min()]
        else:
            fresh=ids[seconds-self.last[ids]>=memory_seconds]
            if len(fresh):ids=fresh
            else:ids=ids[self.last[ids]==self.last[ids].min()]
        return self.centres[ids[np.linalg.norm(self.centres[ids]-position,axis=1).argmin()]].copy()


def route_details(world,position,goal):
    nav=navigation(world);distance=float(np.linalg.norm(goal-position))
    if nav.clear_segment(position,goal):return distance,np.asarray(goal).copy()
    a,b=nav.anchor(position),nav.anchor(goal)
    if a is None or b is None:return None,None
    field,_=nav.field(b)
    if not np.isfinite(field[a]):return None,None
    length=float(field[a]+np.linalg.norm(position-nav.sites[a])+np.linalg.norm(goal-nav.sites[b]))
    return length,nav.waypoint(position,goal)


def observe(world,prey,predator,pellets,available,hunger,velocity,memory,seconds,target_age,stuck_time,config):
    """Only visible object centres; global static terrain and own memory known."""
    nav=navigation(world);objects=[]
    for index in np.flatnonzero(available):
        if nav.visible(prey,pellets[index],config['prey_vision']):
            length,waypoint=route_details(world,prey,pellets[index])
            objects.append(dict(type='food',id=int(index),position=pellets[index].tolist(),relative_position=(pellets[index]-prey).tolist(),distance=float(np.linalg.norm(pellets[index]-prey)),route_distance=length,waypoint=None if waypoint is None else waypoint.tolist()))
    if config['predator'] and nav.visible(prey,predator,config['prey_vision']):
        length,waypoint=route_details(world,prey,predator)
        objects.append(dict(type='predator',id=0,position=predator.tolist(),relative_position=(predator-prey).tolist(),distance=float(np.linalg.norm(predator-prey)),route_distance=length,waypoint=None if waypoint is None else waypoint.tolist()))
    food=[o for o in objects if o['type']=='food'];pred=[o for o in objects if o['type']=='predator']
    features=dict(hunger=hunger/config['maximum_hunger'],food_count=float(len(food)),predator_visible=float(bool(pred)),
                  food_distance=min([o['distance'] for o in food],default=96.),predator_distance=pred[0]['distance'] if pred else 96.,
                  food_route_distance=min([o['route_distance'] for o in food if o['route_distance'] is not None],default=192.),
                  predator_route_distance=pred[0]['route_distance'] if pred and pred[0]['route_distance'] is not None else 192.,
                  x=prey[0]/96,y=prey[1]/96,vx=velocity[0]/config['prey_speed'],vy=velocity[1]/config['prey_speed'],
                  explored_fraction=float(np.mean(seconds-memory.last[memory.eligible]<config['memory_seconds'])),target_age=target_age,stuck_time=stuck_time)
    return dict(features=features,objects=objects)


def observe_predator(world,predator,prey,velocity,prey_velocity,memory,seconds,target_age,stuck_time,target,config):
    seen=navigation(world).visible(predator,prey,config['predator_vision']);objects=[]
    length,waypoint=route_details(world,predator,prey) if seen else (None,None)
    if seen:objects.append(dict(type='prey',id=0,position=prey.tolist(),relative_position=(prey-predator).tolist(),distance=float(np.linalg.norm(prey-predator)),route_distance=length,waypoint=None if waypoint is None else waypoint.tolist()))
    relative=(prey-predator)/96 if seen else np.zeros(2)
    features=dict(prey_visible=float(seen),prey_distance=float(np.linalg.norm(prey-predator)) if seen else 96.,prey_route_distance=length if length is not None else 192.,
                  prey_dx=relative[0],prey_dy=relative[1],prey_vx=prey_velocity[0]/config['prey_speed'] if seen else 0.,prey_vy=prey_velocity[1]/config['prey_speed'] if seen else 0.,
                  x=predator[0]/96,y=predator[1]/96,vx=velocity[0]/config['predator_speed'],vy=velocity[1]/config['predator_speed'],
                  explored_fraction=float(np.mean(seconds-memory.last[memory.eligible]<config['memory_seconds'])),target_age=target_age,stuck_time=stuck_time,target_distance=float(np.linalg.norm(target-predator)))
    return dict(features=features,objects=objects)


def destination(chosen,observation,world,position,memory,seconds,rng,config):
    if chosen['action']=='stay':return position.copy()
    if chosen['action']=='location':return memory.location(position,seconds,chosen['selector'],rng,config['memory_seconds'])
    candidates=[o for o in observation['objects'] if o['type']==chosen['object'] and o['route_distance'] is not None]
    candidates.sort(key=lambda o:(o[chosen['order']],o['id']))
    if not candidates:return position.copy() # No automatic fallback behavior.
    obj=candidates[min(chosen['rank'],len(candidates)-1)];point=np.asarray(obj['position'])
    return point if chosen['action']=='approach' else flee_waypoint(world[0],position,point)


def hunger_events(hunger,events,dt,config):
    """Resolve exact within-step starvation, captures, and pellet pickups."""
    elapsed=0.;collected=[]
    for moment,kind,index in sorted(events,key=lambda e:(e[0],0 if e[1]=='capture' else 1,e[2])):
        if moment>dt+1e-10:continue
        needed=config['metabolism']*(moment-elapsed)
        if hunger<=needed+1e-12:return elapsed+hunger/config['metabolism'],'starved',0.,collected
        hunger-=needed;elapsed=moment
        if kind=='capture':return elapsed,'captured',hunger,collected
        hunger=min(config['maximum_hunger'],hunger+config['pellet_energy']);collected.append(index)
    remaining=dt-elapsed
    if hunger<=config['metabolism']*remaining+1e-12:return elapsed+hunger/config['metabolism'],'starved',0.,collected
    return dt,None,hunger-config['metabolism']*remaining,collected


def episode(tree,scenario,world,config=None,trace=None,predator_tree=None):
    steps=episode_steps(tree,scenario,world,config,trace,predator_tree=predator_tree)
    while True:
        try:next(steps)
        except StopIteration as finished:return finished.value


def episode_steps(tree,scenario,world,config=None,trace=None,emit=False,predator_tree=None):
    """The same scored episode, exposed one physics tick at a time for viewing."""
    config=dict(DEFAULTS,**(config or {}));rng=np.random.default_rng(int(scenario))
    collider,sites,_=world;nav=navigation(world)
    prey=sites[rng.integers(len(sites))].copy()
    candidates=sites[(np.linalg.norm(sites-prey,axis=1)>=24)&(np.linalg.norm(sites-prey,axis=1)<=48)]
    if not len(candidates):raise ValueError('Map lacks predator spawn space 24–48 cells from prey')
    predator=candidates[rng.integers(len(candidates))].copy()
    pellets=[]
    for _ in range(config['pellets']):
        p=spawn_site(rng,sites,[],0,[prey,predator],collider.radius,pellets)
        if p is None:raise ValueError('Insufficient pellet placement space')
        pellets.append(p)
    pellets=np.array(pellets);available=np.ones(len(pellets),bool)
    memory=VisitMemory(world);pred_memory=VisitMemory(world);memory.mark(prey,0);pred_memory.mark(predator,0)
    velocity=np.zeros(2);hunger=config['initial_hunger'];target=prey.copy();pred_target=predator.copy()
    pred_velocity=np.zeros(2);pred_since=0.;pred_stuck_since=0.;pred_stuck_anchor=predator.copy();pred_last=action('stay')
    selected=action('stay');last_selected=selected;target_since=0.;stuck_since=0.;stuck_anchor=prey.copy();collected=0;death='timeout'
    observation=None;seconds=0.;decision_ticks=max(1,round(config['decision_seconds']/DT))
    waypoints=[predator.copy(),prey.copy()];pred_mode='explore' if config['predator'] else 'disabled'
    def record():
        if trace is None and not emit:return
        def confidence(m):return np.where(np.isfinite(m.last),np.exp2(-np.maximum(0,seconds-m.last)/config['memory_seconds']),0).tolist()
        frame=dict(seconds=float(seconds),prey=prey.tolist(),predator=predator.tolist(),hunger=float(hunger),food=collected,pellets=pellets[available].tolist(),target=target.tolist(),action=json_action(selected),death=death if seconds>=config['seconds'] or death!='timeout' else None,
                   waypoints=[p.tolist() for p in waypoints],predator_target=pred_target.tolist(),memory=confidence(memory),predator_memory=confidence(pred_memory),predator_action=pred_mode)
        if trace is not None:trace.append(frame)
        return frame
    yield record()
    for tick in range(round(config['seconds']/DT)):
        seconds=tick*DT
        if np.linalg.norm(prey-stuck_anchor)>1:stuck_since=seconds;stuck_anchor=prey.copy()
        if np.linalg.norm(predator-pred_stuck_anchor)>1:pred_stuck_since=seconds;pred_stuck_anchor=predator.copy()
        if tick%decision_ticks==0:
            observation=observe(world,prey,predator,pellets,available,hunger,velocity,memory,seconds,seconds-target_since,seconds-stuck_since,config)
            selected=decide(tree,observation['features'])
            keep_location=selected['action']=='location' and selected==last_selected and np.linalg.norm(target-prey)>2.4
            next_target=target if keep_location else destination(selected,observation,world,prey,memory,seconds,rng,config)
            if selected!=last_selected or np.linalg.norm(next_target-target)>1e-8:target_since=seconds
            last_selected=selected;target=next_target
            if config['predator']:
                if predator_tree is not None:
                    obs=observe_predator(world,predator,prey,pred_velocity,velocity,pred_memory,seconds,seconds-pred_since,seconds-pred_stuck_since,pred_target,config)
                    pred_action=decide(predator_tree,obs['features'])
                    keep=pred_action['action']=='continue' or (pred_action['action']=='location' and pred_action==pred_last and np.linalg.norm(pred_target-predator)>2.4)
                    next_pred=pred_target if keep else destination(pred_action,obs,world,predator,pred_memory,seconds,rng,config)
                    if pred_action!=pred_last or np.linalg.norm(next_pred-pred_target)>1e-8:pred_since=seconds
                    pred_last=pred_action;pred_target=next_pred;pred_mode=json_action(pred_action)
                elif nav.visible(predator,prey,config['predator_vision']):pred_target=prey.copy();pred_mode='pursue'
                elif np.linalg.norm(pred_target-predator)<2.4:
                    pred_target=pred_memory.location(predator,seconds,'unvisited',rng,config['memory_seconds']);pred_mode='explore'
                elif pred_mode=='pursue':pred_mode='last seen target'
        old=prey.copy();old_pred=predator.copy();paths=[[],[]]
        for side,(position,goal,speed) in enumerate(((predator,pred_target,config['predator_speed'] if config['predator'] else 0),(prey,target,config['prey_speed']))):
            waypoint=nav.waypoint(position,goal);waypoints[side]=waypoint.copy();delta=waypoint-position;length=np.linalg.norm(delta)
            command=delta/max(length,1e-12)*min(speed*DT,length)
            updated,_=collider.move(position,command,path=paths[side])
            if side==0:predator=np.asarray(updated)
            else:prey=np.asarray(updated)
        events=[]
        if config['predator']:
            contact=capture_fraction(paths[0],paths[1],2*collider.radius)
            if contact is not None:events.append((contact*DT,'capture',-1))
        pickups=pickup_times([paths[1]],pellets,np.array([True]),FOOD_RADIUS)[0]
        for index in np.flatnonzero(available & np.isfinite(pickups)):events.append((pickups[index]*DT,'food',int(index)))
        duration,reason,hunger,picked=hunger_events(hunger,events,DT,config)
        collected+=len(picked);available[picked]=False
        if reason:
            prey=interpolate_path(paths[1],[duration/DT])[0];predator=interpolate_path(paths[0],[duration/DT])[0]
            seconds+=duration;death=reason;yield record();break
        # Food respawns, but prey never respawn in a scored first life.
        for index in np.flatnonzero(~available):
            point=spawn_food(rng,sites,[prey],[prey,predator],collider.radius,pellets[available],dict(food_spawn_mode=config.get('food_spawn_mode','soft'),food_spawn_scale=config.get('food_spawn_scale',12)))
            if point is not None:pellets[index]=point;available[index]=True
        seconds=(tick+1)*DT;velocity=(prey-old)/DT;pred_velocity=(predator-old_pred)/DT;memory.mark(prey,seconds);pred_memory.mark(predator,seconds);yield record()
    return dict(survival=float(seconds),food=int(collected),death=death,final_hunger=float(hunger))


def viewer_steps(tree,scenario,world,config,predator_tree=None):
    """Adapt tree episode snapshots to the existing desktop viewer interface."""
    steps=episode_steps(tree,scenario,world,config,emit=True,predator_tree=predator_tree)
    while True:
        try:f=next(steps)
        except StopIteration as finished:return finished.value
        alive=f['death'] not in ('captured','starved')
        yield dict(seconds=f['seconds'],positions=[f['predator'],f['prey']],active=[alive],
                   goals=f['waypoints'],targets=[f['predator_target'],f['target']],
                   modes=[f['predator_action'],f['action']],remaining=f['pellets'],
                   captures=int(f['death']=='captured'),food=[f['food']],memory=[f['memory']],predator_memory=f['predator_memory'],events=[],
                   hunger=f['hunger'],maximum_hunger=config['maximum_hunger'],death=f['death'],
                   predator_enabled=config['predator'],first_life=True)
