"""Counterfactual tests of prey responses to predator information; no training."""
import argparse,csv,json,time
from itertools import product
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.simulation.vision_pilot import observe,network,episodes,setup,SPEED,seed_mean_interval


def velocity(genomes,inputs):
    command=network(genomes,inputs)
    return command*SPEED/np.maximum(1,np.linalg.norm(command,axis=1,keepdims=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);parser.add_argument('--reference',type=Path,required=True);args=parser.parse_args()
    out=args.run/'prey-sensing-check';out.mkdir(exist_ok=False);started=time.perf_counter()
    designs={m:json.loads((args.run/m/'design.json').read_text()) for m in ('coevolution','fixed')};seeds=designs['coevolution']['seeds']
    world=setup(json.loads((args.run/'coevolution/terrain.json').read_text()));rays=world[2]
    if any(world[0].solid):raise ValueError('This diagnostic expects an empty arena')
    cases=list(product((8,16,23,25),(0,1),(0,9),range(8),range(16)))
    distance,moving,pred_speed,food_angle,pred_angle=np.array(cases).T
    food_unit=np.c_[np.cos(food_angle*np.pi/4),np.sin(food_angle*np.pi/4)]
    pred_unit=np.c_[np.cos(pred_angle*np.pi/8),np.sin(pred_angle*np.pi/8)]
    pos=np.full((len(cases),2,2),48.);pos[:,0]+=pred_unit*distance[:,None]
    motion=np.zeros_like(pos);motion[:,0]=-pred_unit*pred_speed[:,None];motion[:,1]=food_unit*(moving*9)[:,None]
    pellets=(pos[:,1]+16*food_unit)[:,None,:];available=np.ones((len(cases),1),bool)
    inputs,visible=observe(pos,motion,(24,24),rays,pellets,available,'unit');masked=inputs[1].copy();masked[:,:5]=0
    np.testing.assert_array_equal(inputs[1][distance==25],masked[distance==25])
    for inputs_index in range(5,18):np.testing.assert_array_equal(inputs[1][:,inputs_index],masked[:,inputs_index])
    # This panel exactly matches the completed 10-seed benchmark.
    panel=[]
    for mode in ('coevolution','fixed'):
        d=json.loads((args.reference/mode/'design.json').read_text());h=json.loads((args.reference/mode/'history.json').read_text())
        panel += [np.array(h[f'equal/{s}/{d["generations"]}'])[0] for s in range(d['seeds'])]
    cached=list(csv.DictReader((args.run/'fixed/opponent-comparison/measurements.csv').open()))
    probes=[];matches=[];summary={}
    for mode in ('coevolution','fixed'):
        champions=json.loads((args.run/mode/'champions.json').read_text());label='Co-evolution' if mode=='coevolution' else 'Fixed opponent'
        for seed in range(seeds):
            prey=np.array(champions[f'equal/{seed}'][1]);genomes=np.repeat(prey[None],len(cases),axis=0)
            enabled_v=velocity(genomes,inputs[1]);masked_v=velocity(genomes,masked)
            radial=np.sum(enabled_v*pred_unit,axis=1)/SPEED;baseline=np.sum(masked_v*pred_unit,axis=1)/SPEED
            shift=radial-baseline;changed=np.linalg.norm(enabled_v-masked_v,axis=1)/SPEED
            for d in (8,16,23,25):
                for state in ('stationary','all'):
                    select=(distance==d)
                    if state=='stationary':select &= (moving==0)&(pred_speed==0)
                    probes.append(dict(mode=mode,seed=seed,distance=d,state=state,radial_change=float(shift[select].mean()),movement_change=float(changed[select].mean()),approach_enabled=float((radial[select]>.05).mean()),approach_masked=float((baseline[select]>.05).mean()),food_alignment_enabled=float(np.sum(enabled_v[select]*food_unit[select],axis=1).mean()/SPEED),food_alignment_masked=float(np.sum(masked_v[select]*food_unit[select],axis=1).mean()/SPEED)))
            ids=np.arange(24)+80000+seed*100
            a=np.repeat(np.array(panel),len(ids),axis=0);b=np.repeat(prey[None],len(a),axis=0)
            result=episodes(a,b,np.tile(ids,len(panel)),(24,24),world,food_encoding='unit',mask_prey_opponent=True)
            enabled=next(r for r in cached if r['training']==label and int(r['seed'])==seed and int(r['generation'])==50)
            matches.append(dict(mode=mode,seed=seed,enabled_survival=float(enabled['prey_survival']),masked_survival=float(1-result[:,2].mean()),enabled_food=float(enabled['food']),masked_food=float(result[:,6].mean()),enabled_fitness=float(enabled['prey_fitness']),masked_fitness=float(result[:,1].mean())))
            print(f'{mode} | seed {seed+1}/{seeds} | probe and masked benchmark complete | {time.perf_counter()-started:.1f}s',flush=True)
    for name,rows in [('probes',probes),('matches',matches)]:
        with (out/f'{name}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for mode,color in [('coevolution','#ba6954'),('fixed','#188b87')]:
        dist=[8,16,23,25];values=np.array([[next(r['radial_change'] for r in probes if r['mode']==mode and r['seed']==s and r['distance']==d and r['state']=='stationary') for d in dist] for s in range(seeds)])
        mean,low,high=seed_mean_interval(values);axes[0].plot(dist,mean,'o-',color=color,label=mode);axes[0].fill_between(dist,low,high,color=color,alpha=.18)
        group=[r for r in matches if r['mode']==mode];summary[mode]={}
        primary=[r for r in probes if r['mode']==mode and r['distance']==16 and r['state']=='stationary']
        for key in ('radial_change','movement_change','approach_enabled','approach_masked'):
            vals=np.array([r[key] for r in primary])[:,None];m,l,h=seed_mean_interval(vals);summary[mode][key]=dict(mean=float(m[0]),ci95=[float(l[0]),float(h[0])])
        for ax,key,title in zip(axes[1:],('survival','food'),('Survival benefit of predator sensing','Food benefit of predator sensing')):
            delta=np.array([r[f'enabled_{key}']-r[f'masked_{key}'] for r in group]);m,l,h=seed_mean_interval(delta[:,None]);i=0 if mode=='coevolution' else 1
            ax.errorbar(i,m[0],yerr=[[m[0]-l[0]],[h[0]-m[0]]],fmt='o',color=color,capsize=5)
            ax.set(title=title,xticks=[0,1],xticklabels=['Co-evolution','Fixed opponent'],xlim=(-.6,1.6));ax.axhline(0,color='gray',lw=1,ls='--')
            summary[mode][f'{key}_sensing_benefit']=dict(enabled=float(np.mean([r[f'enabled_{key}'] for r in group])),masked=float(np.mean([r[f'masked_{key}'] for r in group])),difference=float(m[0]),ci95=[float(l[0]),float(h[0])])
    axes[0].axhline(0,color='gray',lw=1,ls='--');axes[0].axvline(24,color='gray',lw=1,ls=':');axes[0].set(title='Immediate response to seeing predator',xlabel='Predator distance (vision limit: 24)',ylabel='Change in speed toward predator / max speed');axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Survival fraction: sensing on minus off');axes[2].set_ylabel('Pellets: sensing on minus off')
    fig.suptitle('Does prey sensing cause approach or avoidance? Means and 95% seed-bootstrap CIs\nLeft: negative = sensing steers away. Right: positive = sensing improves performance.')
    fig.savefig(out/'sensing-diagnostic.png',dpi=170);plt.close(fig)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'design.json').write_text(json.dumps(dict(probe='Prey centred at (48,48), fixed visible pellet at distance 16; 8 food bearings x 16 predator bearings; distances 8,16,23,25',states='Stationary prey or prey moving toward food at speed 9; stationary or approaching predator at speed 9. Primary plot uses stationary bodies.',intervention='Mask prey opponent displacement, relative velocity, visibility flag only. Food, own velocity, terrain inputs and physical capture rules unchanged.',benchmark='Same six frozen predators and 24 starts per seed as completed 10-seed benchmark; unmasked metrics reused from that benchmark.',notes=['Controlled probes measure immediate policy response, not intention or a complete escape tactic.','Bootstrap averages within evolutionary seeds first; confidence intervals conditional on benchmark panel.'],seconds=time.perf_counter()-started),indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
