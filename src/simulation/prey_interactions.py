"""Independent target choices and continuous circular-body prey collisions."""
import numpy as np
from src.simulation.vision_pilot import DT
from src.simulation.single_pellet import HYBRID_INSTRUCTIONS


class RandomPreyTargets:
    def __init__(self,ids,count):
        self.streams=[np.random.default_rng(np.random.SeedSequence([int(seed),j,938371])) for seed in ids for j in range(count)]
        n=len(self.streams)
        self.food=np.full(n,-1,int);self.food_position=np.zeros((n,2));self.until=np.zeros(n,int);self.explore=np.full(n,-1,int)

    def reset(self,row):
        self.food[row]=-1;self.until[row]=0;self.explore[row]=-1

    def choose(self,policy,position,pellets,available,tick,active):
        centres,eligible=policy.sites;goals=position.copy();foraging=np.zeros(len(position),bool)
        for row in np.flatnonzero(active):
            distance=np.linalg.norm(pellets[row]-position[row],axis=1)
            visible=available[row] & (distance<=policy.rules['prey_vision'])
            choices=np.flatnonzero(visible)
            if len(choices):
                current=self.food[row]
                valid=current>=0 and visible[current] and np.array_equal(self.food_position[row],pellets[row,current])
                if not valid or tick>=self.until[row]:
                    choices=choices[np.argsort(distance[choices],kind='stable')[:3]]
                    weights=np.exp(-(distance[choices]-distance[choices].min())/6);weights/=weights.sum()
                    current=int(self.streams[row].choice(choices,p=weights));self.food[row]=current
                    self.food_position[row]=pellets[row,current];self.until[row]=tick+round(policy.rules.get('target_commitment_seconds',3)/DT)
                goals[row]=pellets[row,current];foraging[row]=True;self.explore[row]=-1;policy.explore[row]=-1
                continue
            self.food[row]=-1
            selectable=eligible & (policy.tile_cooldown[row]<=tick)
            if not selectable.any():selectable=eligible.copy()
            current=self.explore[row]
            reached=current>=0 and np.linalg.norm(centres[current]-position[row])<=HYBRID_INSTRUCTIONS['goal_reached_distance']
            if current<0 or not selectable[current] or reached:
                candidates=np.flatnonzero(selectable & ~policy.visited[row])
                d=np.linalg.norm(centres-position[row],axis=1)
                if len(candidates):candidates=candidates[np.argsort(d[candidates],kind='stable')[:5]]
                else:
                    candidates=np.flatnonzero(selectable)
                    candidates=candidates[np.argsort(policy.last[row,candidates],kind='stable')[:5]]
                weights=np.exp(-(d[candidates]-d[candidates].min())/12);weights/=weights.sum()
                current=int(self.streams[row].choice(candidates,p=weights));self.explore[row]=current
            policy.explore[row]=current;goals[row]=centres[current]
        return goals,foraging


def contact_time(relative,delta,radius,capture=False):
    c=float(relative@relative-radius*radius);dot=float(relative@delta)
    if c<=1e-10:return 0. if capture or dot < -1e-10 else None
    aa=float(delta@delta)
    if aa<1e-18 or dot>=0:return None
    disc=dot*dot-aa*c
    if disc<0:return None
    t=(-dot-np.sqrt(disc))/aa
    return float(t) if 0<=t<=1 else None


def move_prey(collider,position,displacement,active,predator_path=None):
    """Simultaneous swept contacts, equal-mass inelastic sliding, terrain walls.

    Predator motion is prescribed: capture removes a prey at contact, so it
    cannot obstruct other prey later in the timestep. Returned paths include
    each contact for capture/food auditing; no endpoint overlap repair occurs.
    """
    from src.training.hybrid_pursuit import interpolate_path
    pos=np.array(position,float,copy=True);velocity=np.array(displacement,float,copy=True)
    alive=active.copy();velocity[~alive]=0
    paths=[[(0.,p.copy())] for p in pos];hits=np.full(len(pos),np.nan)
    radius=2*collider.radius;now=0.;contacts=0;budget=128+8*len(pos)
    for i in np.flatnonzero(alive):
        if not collider.clear(pos[i]):raise ValueError('Prey starts inside terrain')
        for j in np.flatnonzero(alive & (np.arange(len(pos))>i)):
            if np.linalg.norm(pos[i]-pos[j])<radius-1e-7:raise ValueError('Prey starts overlapping another prey')
    while now<1-1e-12:
        velocity[np.abs(velocity)<1e-9]=0
        boundary=1.
        if predator_path is not None:
            boundary=min(t for t,_ in predator_path if t>now+1e-12)
            pred_start,pred_end=interpolate_path(predator_path,[now,boundary])
        interval=boundary-now;best=1.;event=None
        for i in np.flatnonzero(alive):
            delta=velocity[i]*interval
            length=float(np.linalg.norm(delta))
            # Query from just behind the centre so an already-touching wall
            # still blocks motion despite floating-point contact tolerances.
            if length>1e-12:
                epsilon=1e-5;direction=delta/length
                fraction,normal=collider.sweep(pos[i]-epsilon*direction,delta+epsilon*direction)
                if normal is not None:fraction=max(0.,min(1.,(fraction*(length+epsilon)-epsilon)/length))
            else:fraction,normal=1.,None
            if normal is not None and fraction<=best:best=fraction;event=('wall',i,normal)
            if predator_path is not None:
                fraction=contact_time(pos[i]-pred_start,velocity[i]*interval-(pred_end-pred_start),radius,True)
                if fraction is not None and fraction<=best:best=fraction;event=('capture',i,None)
            for j in np.flatnonzero(alive & (np.arange(len(pos))>i)):
                fraction=contact_time(pos[i]-pos[j],(velocity[i]-velocity[j])*interval,radius)
                if fraction is not None and fraction<=best:best=fraction;event=('prey',i,j)
        # Keep a microscopic clearance at contacts, as the terrain solver does.
        speed=float(np.linalg.norm(velocity,axis=1).max(initial=0))
        travel=best if event is None else max(0.,best-1e-6/max(speed*interval,1e-12))
        pos+=velocity*(interval*travel);now+=interval*best
        # The arena has exact bounds, unlike rounded rock clearances. Remove
        # only roundoff-sized boundary drift after iterative contact impulses.
        bounded=np.clip(pos,collider.radius,collider.size-collider.radius)
        if np.max(np.abs(bounded[active]-pos[active]),initial=0)>1e-8:
            raise RuntimeError('Prey escaped the arena')
        pos[active]=bounded[active]
        for i in range(len(pos)):
            if now>paths[i][-1][0]+1e-12:paths[i].append((now,pos[i].copy()))
            else:paths[i][-1]=(now,pos[i].copy())
        if event is None:continue
        kind,i,other=event
        if kind=='capture':alive[i]=False;velocity[i]=0;hits[i]=now
        elif kind=='wall':
            normal=np.asarray(other);velocity[i]-=min(0.,float(velocity[i]@normal))*normal
        else:
            normal=pos[i]-pos[other];normal/=max(np.linalg.norm(normal),1e-12)
            inward=min(0.,float((velocity[i]-velocity[other])@normal))
            velocity[i]-=.5*inward*normal;velocity[other]+=.5*inward*normal
        contacts+=1
        if contacts>=budget:
            # A jam stops movement; still process remaining predator captures.
            velocity[:]=0
    for i in range(len(pos)):
        if paths[i][-1][0]<1:paths[i].append((1.,pos[i].copy()))
    for i in np.flatnonzero(active):
        if not collider.clear(pos[i]):raise RuntimeError('Prey collision solver entered terrain')
        for j in np.flatnonzero(active & (np.arange(len(pos))>i)):
            # Captured bodies are removed at contact and need not remain apart.
            if np.isnan(hits[i]) and np.isnan(hits[j]) and np.linalg.norm(pos[i]-pos[j])<radius-1e-6:
                raise RuntimeError('Prey collision solver left overlapping bodies')
    return pos,paths,hits,contacts


def pickup_times(paths,pellets,eligible,radius):
    """Earliest pellet contact on actual piecewise paths, not endpoint chords."""
    times=np.full((len(paths),len(pellets)),np.inf)
    for j in np.flatnonzero(eligible):
        for (ta,a),(tb,b) in zip(paths[j],paths[j][1:]):
            delta=np.asarray(b)-a;relative=np.asarray(a)-pellets
            aa=float(delta@delta);bb=relative@delta;cc=np.sum(relative*relative,axis=1)-radius**2
            disc=bb*bb-aa*cc
            t=np.where(cc<=0,0,(-bb-np.sqrt(np.maximum(disc,0)))/max(aa,1e-20))
            valid=((cc<=0)|(aa>1e-18))&(disc>=0)&(t>=0)&(t<=1)
            times[j]=np.minimum(times[j],np.where(valid,ta+t*(tb-ta),np.inf))
    return times
