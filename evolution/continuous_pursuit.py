"""Continuous multi-prey encounters; shared prey genes, private individual memory."""
import numpy as np
from hybrid_pursuit import OrganismPolicy,capture_fraction,interpolate_path
from vision_pilot import DT,FOOD_RADIUS,observe
from terrain_navigation import navigation
from prey_interactions import RandomPreyTargets,move_prey,pickup_times


def spawn_site(rng,sites,observers,radius,bodies,body_radius,occupied=(),spacing=6.):
    """Strict radial exclusion even behind walls; no legal site means wait."""
    valid=np.ones(len(sites),bool)
    for point in observers:valid &= np.linalg.norm(sites-point,axis=1)>radius
    for point in bodies:valid &= np.linalg.norm(sites-point,axis=1)>2*body_radius+.1
    for point in occupied:valid &= np.linalg.norm(sites-point,axis=1)>=spacing
    choices=np.flatnonzero(valid)
    return sites[rng.choice(choices)].copy() if len(choices) else None


def food_spawn_weights(sites,prey,scale=12.):
    """Positive everywhere: a smooth preference for distance, not a vision cutoff."""
    if scale<=0:raise ValueError('Food spawn distance scale must be positive')
    if not len(prey):return np.ones(len(sites))
    distance=np.linalg.norm(np.asarray(sites)[:,None]-np.asarray(prey)[None],axis=2).min(axis=1)
    return .05+.95*(-np.expm1(-.5*(distance/scale)**2))


def spawn_food(rng,sites,prey,bodies,body_radius,occupied,rules):
    if rules.get('food_spawn_mode','strict')=='strict':
        return spawn_site(rng,sites,prey,rules['prey_vision'],bodies,body_radius,occupied)
    weights=food_spawn_weights(sites,prey,rules.get('food_spawn_scale',12.))
    for point in bodies:weights[np.linalg.norm(sites-point,axis=1)<=2*body_radius+.1]=0
    for point in occupied:weights[np.linalg.norm(sites-point,axis=1)<6]=0
    total=weights.sum()
    return sites[rng.choice(len(sites),p=weights/total)].copy() if total>0 else None


def reset_individual(policy,row,position,tick):
    policy.visited[row]=False;policy.last[row]=-1;policy.explore[row]=-1
    policy.history[row]=position;policy.until[row]=tick;policy.next_check[row]=tick+policy.window
    policy.escape[row]=position;policy.recovery_avoids[row]=False;policy.avoiding[row]=False
    policy.tile_cooldown[row]=0
    if policy.food_cooldown is not None:policy.food_cooldown[row]=0
    cell=np.clip((position*policy.grid/96).astype(int),0,policy.grid-1)
    index=cell[1]*policy.grid+cell[0];policy.visited[row,index]=True;policy.last[row,index]=tick
    policy.goals[row]=position
    if policy.target_selector is not None:policy.target_selector.reset(row)


def continuous_episodes(predators,prey_genomes,ids,world,rules,seconds,pellet_count,grid,trace,scripted,progress):
    steps=continuous_steps(predators,prey_genomes,ids,world,rules,seconds,pellet_count,grid,trace,scripted,progress)
    while True:
        try:next(steps)
        except StopIteration as finished:return finished.value


def continuous_steps(predators,prey_genomes,ids,world,rules,seconds,pellet_count=12,grid=8,trace=None,scripted=False,progress=None,emit=False):
    """Advance the actual encounter one physics tick at a time for live viewers.

    Training consumes the same generator without constructing frame payloads.
    The return value is the original metrics array; emitted frames are read-only
    snapshots, so visual toggles cannot affect the simulation.
    """
    n=len(ids);k=rules['prey_count'];collider,sites,rays=world;nav=navigation(world)
    rngs=[np.random.default_rng(int(i)+480000003) for i in ids]
    pos=np.zeros((n,k+1,2));food=np.zeros((n,pellet_count,2));available=np.zeros((n,pellet_count),bool)
    active=np.zeros((n,k),bool);due=np.zeros((n,k),int);food_due=np.zeros((n,pellet_count),int)
    for row,rng in enumerate(rngs):
        pos[row,0]=sites[rng.integers(len(sites))]
        for j in range(k):
            point=spawn_site(rng,sites,[pos[row,0]],rules['predator_vision'],pos[row,:j+1],collider.radius)
            if point is not None:pos[row,j+1]=point;active[row,j]=True
            else:pos[row,j+1]=pos[row,0] # inactive placeholder, never sensed or moved
        for index in range(pellet_count):
            point=spawn_food(rng,sites,pos[row,1:][active[row]],pos[row,np.r_[True,active[row]]],collider.radius,food[row,available[row]],rules)
            if point is not None:food[row,index]=point;available[row,index]=True
    velocity=np.zeros_like(pos)
    pp=OrganismPolicy(0,pos[:,0],world,rules,grid)
    qp=OrganismPolicy(1,pos[:,1:].reshape(-1,2),world,rules,grid)
    qp.visited[~active.ravel()]=False;qp.last[~active.ravel()]=-1
    if rules.get('random_prey_targets',False):qp.target_selector=RandomPreyTargets(ids,k)
    body_solver_contacts=np.zeros(n,int)
    prey_genes=np.repeat(prey_genomes,k,axis=0)
    captures=np.zeros(n);collected=np.zeros((n,k));alive_time=np.zeros((n,k));detected=np.zeros((n,k));avoiding=np.zeros((n,k));proximity=np.zeros(n)
    delay=max(1,round(rules.get('respawn_delay',1)/DT));steps=round(seconds/DT)
    events=[]
    def record(tick):
        if trace is None and not emit:return
        def memory(policy,rows):
            age=np.maximum(0,(tick-policy.last[rows])*DT)
            confidence=np.where(policy.last[rows]>=0,np.exp2(-age/rules['memory_half_life']),0)
            return confidence.tolist()
        frame=dict(seconds=tick*DT,positions=pos[0].tolist(),active=active[0].tolist(),goals=[pp.goals[0].tolist()]+qp.goals[:k].tolist(),targets=[getattr(pp,'targets',pp.goals)[0].tolist()]+getattr(qp,'targets',qp.goals)[:k].tolist(),modes=[str(pp.modes[0])]+[str(qp.modes[j]) if active[0,j] else 'respawn pending' for j in range(k)],remaining=food[0,available[0]].tolist(),captures=int(captures[0]),body_solver_contacts=int(body_solver_contacts[0]),food=collected[0].tolist(),memory=memory(qp,slice(0,k)),predator_memory=memory(pp,slice(0,1))[0],events=list(events))
        if trace is not None:trace.append(frame)
        return frame
    yield record(0)
    for tick in range(steps):
        events=[]
        if progress is not None and tick and tick%50==0:progress(tick*DT,seconds,int(active.sum()),n*k)
        # Retry failed placements without weakening exclusion constraints.
        for row,j in np.argwhere(~active & (due<=tick)):
            point=spawn_site(rngs[row],sites,[pos[row,0]],rules['predator_vision'],pos[row,np.r_[True,active[row]]],collider.radius)
            due[row,j]=tick+delay
            if point is not None:
                pos[row,j+1]=point;velocity[row,j+1]=0;active[row,j]=True
                reset_individual(qp,row*k+j,point,tick)
                if row==0:events.append(dict(type='prey_respawn',prey=int(j),position=point.tolist(),predator_position=pos[row,0].tolist()))
        for row,index in np.argwhere(~available & (food_due<=tick)):
            point=spawn_food(rngs[row],sites,pos[row,1:][active[row]],pos[row,np.r_[True,active[row]]],collider.radius,food[row,available[row]],rules)
            food_due[row,index]=tick+delay
            if point is not None:
                food[row,index]=point;available[row,index]=True
                if qp.food_cooldown is not None:qp.food_cooldown[row*k:(row+1)*k,index]=0
                if row==0:events.append(dict(type='food_respawn',position=point.tolist(),prey_positions=pos[row,1:][active[row]].tolist()))
        # Predator selects the nearest actually visible prey. Dummy target is unseen.
        distance=np.linalg.norm(pos[:,1:]-pos[:,0,None],axis=-1)
        visible=active & (distance<=rules['predator_vision'])
        if rules.get('vision_occlusion',False):
            for row,j in np.argwhere(visible):visible[row,j]=nav.visible(pos[row,0],pos[row,j+1],rules['predator_vision'])
        chosen=np.where(visible,distance,np.inf).argmin(axis=1);has=visible.any(axis=1)
        target=pos[np.arange(n),chosen+1].copy();target[~has]=pos[~has,0]+100000
        target_vel=velocity[np.arange(n),chosen+1].copy();target_vel[~has]=0
        inputs,_=observe(np.stack((pos[:,0],target),axis=1),np.stack((velocity[:,0],target_vel),axis=1),(rules['predator_vision'],rules['prey_vision']),rays)
        desired_pred=pp.command(predators,pos[:,0],target,inputs[0],food,available,tick,np.ones(n,bool),scripted)
        flatpos=pos[:,1:].reshape(-1,2);flatpred=np.repeat(pos[:,0],k,axis=0)
        qinputs,_=observe(np.stack((flatpred,flatpos),axis=1),np.stack((np.repeat(velocity[:,0],k,axis=0),velocity[:,1:].reshape(-1,2)),axis=1),(rules['predator_vision'],rules['prey_vision']),rays)
        desired_prey=qp.command(prey_genes,flatpos,flatpred,qinputs[1],np.repeat(food,k,axis=0),np.repeat(available,k,axis=0),tick,active.ravel(),scripted).reshape(n,k,2)
        old=pos.copy();was_active=active.copy();fractions=np.ones((n,k));tag=np.zeros((n,k),bool)
        for row in range(n):
            near=active[row] & (distance[row]<=2*collider.radius+2*rules['speed']*DT)
            pred_path=[] if near.any() else None
            pos[row,0]=collider.move(pos[row,0],desired_pred[row]*DT,path=pred_path)[0]
            paths=None
            if rules.get('prey_collisions',False):
                pos[row,1:],paths,hits,contacts=move_prey(collider,pos[row,1:],desired_prey[row]*DT,active[row],pred_path)
                body_solver_contacts[row]+=contacts
                tag[row]=np.isfinite(hits);fractions[row,tag[row]]=hits[tag[row]]
                if row==0:
                    for j in np.flatnonzero(tag[row]):events.append(dict(type='capture',prey=int(j),seconds=(tick+hits[j])*DT))
            else:
                for j in np.flatnonzero(active[row]):
                    prey_path=[] if near[j] else None
                    pos[row,j+1]=collider.move(pos[row,j+1],desired_prey[row,j]*DT,path=prey_path)[0]
                    if near[j]:
                        hit=capture_fraction(pred_path,prey_path,2*collider.radius)
                        if hit is not None:
                            tag[row,j]=True;fractions[row,j]=hit
                            pos[row,j+1]=interpolate_path(prey_path,[hit])[0]
                            if row==0:events.append(dict(type='capture',prey=int(j),seconds=(tick+hit)*DT))
            # Competing prey can claim each pellet only once. Earliest swept
            # pickup wins; ties break by prey index. No pickup after capture.
            pickup_matrix=pickup_times(paths,food[row],was_active[row]&~tag[row],FOOD_RADIUS) if paths is not None else None
            for index in np.flatnonzero(available[row]):
                delta=pos[row,1:]-old[row,1:];relative=old[row,1:]-food[row,index]
                aa=np.sum(delta*delta,axis=1);bb=np.sum(relative*delta,axis=1);cc=np.sum(relative*relative,axis=1)-FOOD_RADIUS**2
                disc=bb*bb-aa*cc;pickup=np.where(cc<=0,0,(-bb-np.sqrt(np.maximum(disc,0)))/np.maximum(aa,1e-20))
                valid=was_active[row]&~tag[row]&((cc<=0)|(aa>1e-18))&(disc>=0)&(pickup>=0)&(pickup<=1)
                if pickup_matrix is not None:
                    pickup=pickup_matrix[:,index];valid=np.isfinite(pickup)
                if valid.any():
                    winner=np.where(valid,pickup,np.inf).argmin();collected[row,winner]+=1
                    available[row,index]=False;food_due[row,index]=tick+1+delay
        velocity=(pos-old)/DT
        captures+=tag.sum(axis=1);active[tag]=False;due[tag]=tick+1+delay
        weight=was_active*fractions*DT;alive_time+=weight
        detected+=visible*weight;avoiding+=qp.avoiding.reshape(n,k)*weight
        nearest=np.where(was_active,distance,np.inf).min(axis=1);proximity+=np.exp(-nearest/12)*DT/seconds
        pp.mark(pos[:,0],tick+1,np.ones(n,bool));qp.mark(pos[:,1:].reshape(-1,2),tick+1,was_active.ravel())
        yield record(tick+1)
    alive=alive_time.sum(axis=1)/(k*seconds);food_mean=collected.mean(axis=1)
    return np.c_[captures+.2*proximity,alive-captures/k+.25*food_mean,captures,alive,food_mean,detected.sum(axis=1)/np.maximum(alive_time.sum(axis=1),DT),avoiding.sum(axis=1)/np.maximum(alive_time.sum(axis=1),DT),pp.visited.sum(axis=1),qp.visited.reshape(n,k,-1).sum(axis=2).mean(axis=1)]


def render_continuous(terrain,examples,rules,seconds,out):
    from PIL import Image,ImageDraw,ImageFont
    from grid_collision import GridCollider
    from terrain_navigation import TerrainNavigation
    scale=5;size=480;top=86;gap=12
    solid=np.array(terrain['solid'],bool).reshape(96,96)
    bg=Image.fromarray(np.where(solid[:,:,None],[108,119,116],[232,228,206]).astype('uint8')).resize((size,size),Image.Resampling.NEAREST).convert('RGBA')
    sight=TerrainNavigation.__new__(TerrainNavigation);sight.sight=GridCollider(terrain,radius=0)
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',15);frames=[]
    # Show one encounter at readable scale; all three raw replays are saved.
    for tick in range(0,len(examples[0]),2):
        state=examples[0][tick];image=bg.copy();overlay=Image.new('RGBA',image.size)
        d=ImageDraw.Draw(overlay)
        for cell,confidence in enumerate(state['memory'][0]):
            x,y=cell%8,cell//8;d.rectangle((x*60,y*60,(x+1)*60-1,(y+1)*60-1),fill=(20,170,80,round(100*confidence)))
        mask=np.array(overlay);mask[np.repeat(np.repeat(solid,scale,axis=0),scale,axis=1),3]=0
        image=Image.alpha_composite(image,Image.fromarray(mask));d=ImageDraw.Draw(image)
        for line in range(9):
            q=min(size-1,line*60);d.line((q,0,q,size-1),fill='#9da995');d.line((0,q,size-1,q),fill='#9da995')
        for x,y in state['remaining']:
            x*=scale;y*=scale;d.ellipse((x-4,y-4,x+4,y+4),fill='#e3b72f',outline='#66551a')
        colors=['#df755e','#239b9d','#3972bf','#aa5cbb','#8e9237']
        for j,point in enumerate(state['positions']):
            if j and not state['active'][j-1]:continue
            color=colors[j%len(colors)];radius=rules['predator_vision' if j==0 else 'prey_vision']
            if rules.get('vision_occlusion',False):outline=[tuple(p*scale) for p in sight.vision_outline(point,radius)]
            else:outline=[tuple((np.asarray(point)+radius*np.array([np.cos(a),np.sin(a)]))*scale) for a in np.arange(64)*np.pi/32]
            d.line(outline+[outline[0]],fill=color,width=1)
            x,y=np.array(point)*scale;gx,gy=np.array(state['goals'][j])*scale
            d.line((x,y,gx,gy),fill=color);r=2.4*scale;d.ellipse((x-r,y-r,x+r,y+r),fill=color,outline='white',width=2)
            d.text((x+12,y-12),'P' if j==0 else str(j),font=font,fill='#14201d')
        frame=Image.new('RGB',(size+2*gap,top+size+100),'#172321');draw=ImageDraw.Draw(frame)
        draw.text((gap,8),f'Continuous pursuit | {rules["prey_count"]} prey | playback 2x',font=font,fill='white')
        draw.text((gap,32),f'{state["seconds"]:.1f}s | captures {state["captures"]} | food {sum(state["food"]):g}',font=font,fill='white')
        draw.text((gap,56),f'Green: prey 1 memory | half-life {rules["memory_half_life"]:g}s',font=font,fill='white')
        frame.paste(image.convert('RGB'),(gap,top))
        draw.text((gap,top+size+8),'Predator: '+state['modes'][0],font=font,fill='white')
        for j,mode in enumerate(state['modes'][1:]):
            if j<3:draw.text((gap,top+size+30+j*20),f'Prey {j+1}: {mode}',font=font,fill='white')
        frames.append(frame)
    frames[0].save(out/'replay.gif',save_all=True,append_images=frames[1:],duration=100,loop=0)
    frames[min(25,len(frames)-1)].save(out/'replay-preview.png')
