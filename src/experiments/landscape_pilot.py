"""Pilot of route connectivity and food concentration during co-evolution."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from src.experiments.ablate_pursuit import GENES, RADIUS, STEPS, DT, episodes, mutate

SETTINGS = [('one', 'clustered'), ('one', 'dispersed'),
            ('three', 'clustered'), ('three', 'dispersed')]


def landscape(connectivity, concentration, seed):
    rng = np.random.default_rng(seed)
    shift = rng.uniform(-.18, .18)
    gaps = np.array([0.]) if connectivity == 'one' else np.array([-2.2, 0, 2.2])
    gaps += shift
    ys = np.arange(-4, 4.01, .5)
    keep = np.min(np.abs(ys[:,None] - gaps), axis=1) > .72
    obstacles = np.c_[np.zeros(keep.sum()), ys[keep], np.full(keep.sum(), .29)]
    pellets = []
    for side in (-1, 1):
        if concentration == 'clustered':
            center = np.array([side * 2.25, rng.uniform(-1.8, 1.8)])
            for i in range(6):
                angle = i * np.pi / 3 + rng.uniform(-.1, .1)
                pellets.append(center + .37 * np.array([np.cos(angle), np.sin(angle)]))
        else:
            for i in range(6):
                pellets.append([side * rng.uniform(1.1, 3.1), -2.8 + i * 1.1 + rng.uniform(-.12,.12)])
    pellets = np.array(pellets)
    rotated = bool(seed % 2)
    if rotated:
        obstacles[:,:2] = obstacles[:,[1,0]]
        pellets = pellets[:,::-1].copy()
    return {'obstacles':obstacles,'pellets':pellets,'rotate':rotated}


def batch(a, b, seed_ids, maps, trace=None):
    values = []
    for environment in maps:
        values.append(episodes(a,b,seed_ids,'turning',environment,trace))
    return np.stack(values)


def background(ax, environment):
    for x,y,r in environment['obstacles']:
        ax.add_patch(Circle((x,y),r,color='#667276'))
    ax.scatter(*environment['pellets'].T,s=23,color='#c8a023',edgecolor='#64531b',linewidth=.3)
    ax.set(xlim=(-4,4),ylim=(-4,4),aspect='equal',xticks=[],yticks=[])
    ax.set_facecolor('#f4f1e7')


def save_figures(out, rows, examples):
    fig, axes = plt.subplots(2,2,figsize=(8,8),layout='constrained')
    for ax, ((connectivity,concentration), example) in zip(axes.flat, zip(SETTINGS,examples)):
        background(ax,example['environment'])
        path = np.asarray(example['path'])[:,0]
        for side,color,label in [(0,'#c44c43','Predator'),(1,'#218c93','Prey')]:
            ax.plot(path[:,side,0],path[:,side,1],color=color,lw=1.4,label=label)
            ax.scatter(*path[0,side],color=color,s=45,marker='o',edgecolor='white',zorder=4)
            ax.scatter(*path[-1,side],color=color,s=55,marker='X',edgecolor='white',zorder=4)
        label = 'One passage' if connectivity == 'one' else 'Three passages'
        ax.set_title(f'{label}, {concentration} food',fontsize=11)
    axes[0,0].legend(loc='upper left',fontsize=8)
    fig.suptitle('Final champions: first seed, first held-out map and start',fontsize=13,fontweight='bold')
    fig.savefig(out/'landscapes-and-paths.png',dpi=175)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11.4,4.4),layout='constrained')
    summary={}
    for ax,metric,title in zip(axes,['capture','food','crossings'],
                               ['Predator capture rate (%)','Pellets collected by prey','Passage crossings by prey']):
        matrix = np.zeros((2,2))
        for i,connectivity in enumerate(('one','three')):
            for j,concentration in enumerate(('clustered','dispersed')):
                group=[r[metric] for r in rows if r['connectivity']==connectivity and r['food_layout']==concentration and r['generation']==16]
                matrix[i,j]=np.mean(group)*(100 if metric=='capture' else 1)
                summary.setdefault(connectivity+'/'+concentration,{})[metric]=float(matrix[i,j])
        ceiling = max(float(matrix.max()),1)
        ax.imshow(matrix,cmap='Blues',vmin=0,vmax=ceiling,aspect='equal')
        for i in range(2):
            for j in range(2):
                text=f'{matrix[i,j]:.0f}%' if metric=='capture' else f'{matrix[i,j]:.2f}'
                ax.text(j,i,text,ha='center',va='center',fontsize=21,
                        color='white' if matrix[i,j]>ceiling*.55 else '#162b3c')
        ax.set(title=title,xticks=[0,1],xticklabels=['Concentrated','Dispersed'],
               yticks=[0,1],yticklabels=['1 passage','3 passages'])
    fig.suptitle('Final co-evolving pairs on held-out maps: means over three seeds',fontweight='bold',fontsize=13)
    fig.savefig(out/'outcomes.png',dpi=175)
    plt.close(fig)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for connectivity,concentration in SETTINGS:
        for ax,key in zip(axes,['capture','food']):
            generations=sorted(set(r['generation'] for r in rows))
            means=[np.mean([r[key] for r in rows if r['connectivity']==connectivity and r['food_layout']==concentration and r['generation']==g]) for g in generations]
            ax.plot(generations,means,'o-',label=f'{connectivity} / {concentration}')
    axes[0].set(title='Contemporary capture rate',xlabel='Generation',ylabel='Fraction caught')
    axes[1].set(title='Prey collection',xlabel='Generation',ylabel='Pellets per episode')
    axes[1].legend(fontsize=8)
    fig.savefig(out/'progress.png',dpi=175)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/landscape-pilot'))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    rows=[];champion_data=[];examples=[]
    for connectivity,concentration in SETTINGS:
        train_maps=[landscape(connectivity,concentration,s) for s in (100,101,102)]
        test_maps=[landscape(connectivity,concentration,s) for s in (200,201,202)]
        for seed in range(3):
            rng=np.random.default_rng(710+seed)
            initial=rng.normal(0,.45,(4,GENES))
            streams=[np.random.default_rng(1200+2*seed+side) for side in range(2)]
            champions=[initial[0],initial[0]]
            populations=[initial,initial]
            train_starts=np.arange(2)+seed*100
            test_starts=np.arange(4)+5000+seed*100
            for generation in range(17):
                if generation:
                    a=np.repeat(populations[0],4*len(train_starts),axis=0)
                    b=np.tile(np.repeat(populations[1],len(train_starts),axis=0),(4,1))
                    results=batch(a,b,np.tile(train_starts,16),train_maps).reshape(3,4,4,2,9)
                    scores=[results[:,:,:,:,0].mean(axis=(0,2,3)),results[:,:,:,:,1].mean(axis=(0,1,3))]
                    champions=[p[s.argmax()].copy() for p,s in zip(populations,scores)]
                    populations=[np.array([g]+[mutate(g,r) for _ in range(3)]) for g,r in zip(champions,streams)]
                if generation in (0,8,16):
                    a=np.repeat(champions[0][None],len(test_starts),axis=0)
                    b=np.repeat(champions[1][None],len(test_starts),axis=0)
                    result=batch(a,b,test_starts,test_maps).reshape(-1,9)
                    for trial,v in enumerate(result):
                        rows.append({'connectivity':connectivity,'food_layout':concentration,'seed':seed,
                            'generation':generation,'map_seed':200+trial//len(test_starts),'start':int(test_starts[trial%len(test_starts)]),
                            'capture':float(v[2]),'survival':float(v[3]),'food':float(v[6]),'crossings':float(v[8])})
            champion_data.append({'connectivity':connectivity,'food_layout':concentration,'seed':seed,
                                  'predator':champions[0].tolist(),'prey':champions[1].tolist()})
            if seed==0:
                path=[]
                episodes(champions[0][None],champions[1][None],[5000],'turning',test_maps[0],path)
                examples.append({'environment':test_maps[0],'path':path})
            print(f'{connectivity} / {concentration} | seed {seed+1}/3 | elapsed {time.perf_counter()-start:.1f}s',flush=True)
            with (args.output/'measurements.csv').open('w',newline='',encoding='utf-8') as file:
                writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (args.output/'champions.json').write_text(json.dumps(champion_data),encoding='utf-8')
    design={'generations':16,'optimizer_seeds':[710,711,712],'population_per_role':4,'training_maps':[100,101,102],
            'held_out_maps':[200,201,202],'duration':STEPS*DT,'training_starts_per_map':2,'test_starts_per_map':4,
            'genes':GENES,'pellets':12,'body_radius':RADIUS,
            'notes':['Simplified NumPy 2D simulator; both turn at 3 rad/s and have speed limit 2.',
                     'More passages remove wall material; passage width is held approximately constant.',
                     'All starts are in the same chamber; both chambers contain six pellets.',
                     'Prey reward includes survival and .25 per pellet; no starvation or regrowth.',
                     'Full visibility of opponent; food sensor points toward nearest available pellet.',
                     'Examples show first seed and first held-out start without selecting a favorable replay.'],
            'wall_seconds':time.perf_counter()-start}
    (args.output/'design.json').write_text(json.dumps(design,indent=2),encoding='utf-8')
    save_figures(args.output,rows,examples)


if __name__=='__main__':
    main()
