"""Cross-evaluate saved hybrid controllers; no additional evolution."""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.training.hybrid_pursuit import episodes, HYBRID_INSTRUCTIONS
from src.simulation.vision_pilot import setup, seed_mean_interval


FIELDS=['seed','predator_generation','prey_generation','start','capture','duration','food']


def plot(rows,out):
    seeds=sorted({int(r['seed']) for r in rows})
    gs=sorted({int(r['predator_generation']) for r in rows})
    design=json.loads((out/'design.json').read_text())
    if seeds!=design['seeds'] or gs!=design['generations']:
        raise ValueError('Historical evaluation is incomplete; wait for all seeds and pairings')
    for seed in seeds:
        expected=set(range(90000000+seed*1000,90000000+seed*1000+design['starts']))
        for a in gs:
            for b in gs:
                starts=[int(r['start']) for r in rows if int(r['seed'])==seed and int(r['predator_generation'])==a and int(r['prey_generation'])==b]
                if len(starts)!=len(expected) or set(starts)!=expected:
                    raise ValueError(f'Incomplete or duplicate encounters for seed {seed}, matchup {a}/{b}')
    def values(metric):
        return np.array([[[np.mean([float(r[metric]) for r in rows if int(r['seed'])==seed and int(r['predator_generation'])==a and int(r['prey_generation'])==b]) for b in gs] for a in gs] for seed in seeds])
    capture=values('capture');food=values('food')
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    ax=axes[0,0];matrix=capture.mean(axis=0)
    im=ax.imshow(matrix,vmin=0,vmax=1,cmap='magma',origin='lower')
    ax.set(xticks=range(len(gs)),xticklabels=gs,yticks=range(len(gs)),yticklabels=gs,xlabel='Prey generation',ylabel='Predator generation',title='Historical matchups: predator capture rate')
    for a in range(len(gs)):
        for b in range(len(gs)):ax.text(b,a,f'{matrix[a,b]:.0%}',ha='center',va='center',color='white' if matrix[a,b]<.65 else 'black')
    fig.colorbar(im,ax=ax,label='Capture fraction')
    panels=[(axes[0,1],capture,'Predator capture against fixed prey','Evolving predator generation','Capture fraction',0),
            (axes[1,0],1-capture,'Prey survival against fixed predators','Evolving prey generation','Survival fraction',1),
            (axes[1,1],food,'Prey food against fixed predators','Evolving prey generation','Pellets per encounter',1)]
    for ax,data,title,xlabel,ylabel,side in panels:
        for index,color in ((0,'#198c87'),(len(gs)-1,'#c46b34')):
            trajectories=data[:,:,index] if side==0 else data[:,index,:]
            mean,low,high=seed_mean_interval(trajectories)
            ax.plot(gs,mean,marker='o',color=color,label=f'Fixed opponent: generation {gs[index]}')
            ax.fill_between(gs,low,high,color=color,alpha=.16)
        ax.set(title=title,xlabel=xlabel,ylabel=ylabel,xticks=gs,ylim=(0,1) if ylabel!='Pellets per encounter' else (0,None))
        ax.legend(fontsize=8)
    fig.suptitle(f'Saved co-evolution controllers evaluated against fixed opponents | {len(seeds)} seeds\nSame starts across every pairing; bands are pointwise 95% bootstrap CIs across seeds')
    fig.savefig(out/'historical-matchups.png',dpi=170);plt.close(fig)
    summary={}
    for label,data in [('predator_capture_vs_initial_prey',capture[:,:,0]),('prey_survival_vs_initial_predator',1-capture[:,0,:]),('prey_food_vs_initial_predator',food[:,0,:])]:
        change=data[:,-1]-data[:,0];mean,lo,hi=seed_mean_interval(change[:,None])
        summary[label]=dict(initial=float(data[:,0].mean()),final=float(data[:,-1].mean()),change=float(mean[0]),change_ci95=[float(lo[0]),float(hi[0])],changes_by_seed=change.tolist())
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--starts',type=int,default=8,help='First saved evaluation starts, reused across all matchups')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--plot-only',action='store_true')
    args=parser.parse_args();out=args.output or args.run/'historical'
    if args.plot_only:
        with (out/'matchups.csv').open(newline='') as f:plot(list(csv.DictReader(f)),out)
        return
    design=json.loads((args.run/'design.json').read_text());instructions=json.loads((args.run/'instructions.json').read_text())
    if design['rules'].get('respawn',False):parser.error('Historical analysis currently supports only single-life runs; continuous runs have different metrics')
    if instructions['foraging_and_recovery']!=HYBRID_INSTRUCTIONS:parser.error('Saved recovery rules differ from current implementation; restore saved rules before evaluation')
    if not 1<=args.starts<=design['test_starts']:parser.error('Starts must be between 1 and the source run test-start count')
    history=json.loads((args.run/'history.json').read_text());seeds=sorted({int(k.split('/')[0]) for k in history})
    gs=sorted({int(k.split('/')[1]) for k in history})
    if any(f'{s}/{g}' not in history for s in seeds for g in gs):parser.error('Run has incomplete saved checkpoints')
    with (args.run/'measurements.csv').open(newline='') as f:old={(int(r['seed']),int(r['generation']),int(r['start'])):r for r in csv.DictReader(f)}
    out.mkdir(parents=True,exist_ok=False)
    (out/'design.json').write_text(json.dumps(dict(source=str(args.run.resolve()),seeds=seeds,generations=gs,starts=args.starts,seconds=design['episode_seconds'],notes=['No retraining; this tests progress of co-evolved controllers against fixed saved opponents, not training against a fixed opponent.','Same saved evaluation starts for all pairings within each seed. Diagonal reuses original measurements.','Three independent controller seeds give limited confidence; starting arrangements are not independent evolution runs.']),indent=2))
    world=setup(json.loads((args.run/'terrain.json').read_text()));rows=[];started=time.perf_counter();done=0;total=len(seeds)*len(gs)**2
    with (out/'matchups.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader()
        for seed in seeds:
            ids=np.arange(args.starts)+90000000+seed*1000
            for a in gs:
                for b in gs:
                    if a==b:
                        measured=[old[(seed,a,int(i))] for i in ids]
                    else:
                        predator=np.array(history[f'{seed}/{a}'][0]);prey=np.array(history[f'{seed}/{b}'][1])
                        result=episodes(np.repeat(predator[None],args.starts,axis=0),np.repeat(prey[None],args.starts,axis=0),ids,world,instructions['pursuit'],seconds=design['episode_seconds'])
                        measured=[dict(capture=r[2],duration=r[3],food=r[4]) for r in result]
                    for i,r in zip(ids,measured):
                        row=dict(seed=seed,predator_generation=a,prey_generation=b,start=int(i),**{k:float(r[k]) for k in ('capture','duration','food')})
                        rows.append(row);writer.writerow(row)
                    f.flush();done+=1
                    print(f'Historical matchups {done}/{total} | seed {seed} | predator {a}, prey {b} | {time.perf_counter()-started:.1f}s'+(' | reused' if a==b else ''),flush=True)
    plot(rows,out)


if __name__=='__main__':main()
