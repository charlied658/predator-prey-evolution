"""Small hunger/survival feasibility experiment with evolved decision trees."""
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.simulation.hunger_tree import DEFAULTS,FEATURES,BASELINE,action,random_tree,mutate,tree_text,episode
from src.simulation.vision_pilot import setup,seed_mean_interval
from src.paths import ROOT


def save_replay(out,terrain,examples,config):
    payload=json.dumps(dict(terrain=terrain,examples=examples,config=config)).replace('</','<\\/')
    template=(ROOT/'src/viewer/hunger_tree_replay.html').read_text(encoding='utf-8')
    (out/'viewer.html').write_text(template.replace('/*PAYLOAD*/',payload),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/hunger-tree-pilot')
    parser.add_argument('--seeds',type=int,default=3);parser.add_argument('--generations',type=int,default=10)
    parser.add_argument('--population',type=int,default=12);parser.add_argument('--training-starts',type=int,default=3)
    parser.add_argument('--test-starts',type=int,default=8);parser.add_argument('--seconds',type=float,default=30)
    parser.add_argument('--eval-every',type=int,default=2);parser.add_argument('--depth',type=int,default=3)
    parser.add_argument('--terrain-map',type=Path);parser.add_argument('--no-predator',action='store_true')
    args=parser.parse_args()
    if min(args.seeds,args.generations,args.training_starts,args.test_starts,args.eval_every,args.seconds,args.depth)<=0 or args.population<4:parser.error('Positive budgets, depth and at least four individuals are required')
    if abs(round(args.seconds/.1)*.1-args.seconds)>1e-8:parser.error('Duration must be a multiple of 0.1 seconds')
    if args.depth>6:parser.error('Use depth at most six for this pilot')
    out=args.output;out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    config=dict(DEFAULTS,seconds=args.seconds,predator=not args.no_predator)
    terrain=json.loads(args.terrain_map.read_text()) if args.terrain_map else dict(resolution=96,solid=[0]*96**2,preset='Empty arena')
    (out/'terrain.json').write_text(json.dumps(terrain))
    world=setup(terrain);rows=[];champions={};initial={};history={};examples={}
    def benchmark(tree,ids,method,seed,generation):
        results=[episode(tree,int(i),world,config) for i in ids]
        for i,r in zip(ids,results):rows.append(dict(method=method,seed=seed,generation=generation,scenario=int(i),**r))
        with (out/'measurements.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        return results
    for seed in range(args.seeds):
        rng=np.random.default_rng(20261121+seed);population=[random_tree(rng,args.depth) for _ in range(args.population)]
        tests=np.arange(args.test_starts)+90000000+seed*1000
        for generation in range(args.generations+1):
            ids=np.arange(args.training_starts)+1000000+seed*10000+generation*100
            scores=np.array([np.mean([episode(tree,int(i),world,config)['survival'] for i in ids]) for tree in population])
            # Survival is the ONLY selection signal; ties break randomly.
            order=np.lexsort((rng.random(len(scores)),-scores));champion=deepcopy(population[order[0]])
            if generation==0:initial[str(seed)]=deepcopy(champion)
            if generation%args.eval_every==0 or generation==args.generations:
                measured=benchmark(champion,tests,'evolved',seed,generation)
                history[f'{seed}/{generation}']=champion
                print(f'Check | seed {seed+1}/{args.seeds} | gen {generation} | held-out survival {np.mean([r["survival"] for r in measured]):.2f}s | food {np.mean([r["food"] for r in measured]):.2f}',flush=True)
            print(f'Train | seed {seed+1}/{args.seeds} | generation {generation}/{args.generations} | train survival {scores.max():.2f}s | elapsed {time.perf_counter()-started:.1f}s',flush=True)
            if generation<args.generations:
                children=[deepcopy(population[i]) for i in order[:2]]
                while len(children)<args.population:
                    contenders=rng.choice(len(population),3,replace=False)
                    winner=int(contenders[np.argmax(scores[contenders])])
                    children.append(mutate(population[winner],rng,args.depth))
                population=children
        champions[str(seed)]=champion
        benchmark(BASELINE,tests,'scripted',seed,args.generations)
        benchmark(action('stay'),tests,'stationary',seed,args.generations)
        (out/'trees.json').write_text(json.dumps(dict(initial=initial,champions=champions,history=history),indent=2))
        (out/f'seed-{seed}-tree.txt').write_text(tree_text(champion),encoding='utf-8')
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    generations=sorted({r['generation'] for r in rows if r['method']=='evolved'})
    for ax,key,label in [(axes[0,0],'survival','Held-out first-life survival (s)'),(axes[0,1],'food','Held-out pellets collected')]:
        values=np.array([[np.mean([r[key] for r in rows if r['method']=='evolved' and r['seed']==s and r['generation']==g]) for g in generations] for s in range(args.seeds)])
        mean,lo,hi=seed_mean_interval(values);ax.plot(generations,mean,marker='o',color='#198c87');ax.fill_between(generations,lo,hi,alpha=.2,color='#198c87')
        for method,color in [('scripted','#c46b35'),('stationary','#777777')]:
            value=np.mean([r[key] for r in rows if r['method']==method]);ax.axhline(value,color=color,ls='--',label=method)
        ax.set(xlabel='Generation',ylabel=label,title=label,ylim=(0,config['seconds']+.5) if key=='survival' else (0,None));ax.legend()
    groups=[('Initial','evolved',0),('Evolved','evolved',args.generations),('Scripted','scripted',args.generations),('Stationary','stationary',args.generations)]
    summary={}
    for label,method,g in groups:
        selected=[r for r in rows if r['method']==method and r['generation']==g]
        values=np.array([[np.mean([r['survival'] for r in selected if r['seed']==s])] for s in range(args.seeds)])
        mean,lo,hi=seed_mean_interval(values)
        summary[label]=dict(survival=float(mean[0]),ci95=[float(lo[0]),float(hi[0])],food=float(np.mean([r['food'] for r in selected])),captured=float(np.mean([r['death']=='captured' for r in selected])),starved=float(np.mean([r['death']=='starved' for r in selected])),timeout=float(np.mean([r['death']=='timeout' for r in selected])),seed_survival=values[:,0].tolist())
    x=np.arange(4);ax=axes[1,0];means=np.array([summary[name]['survival'] for name,_,_ in groups]);low=np.array([summary[name]['ci95'][0] for name,_,_ in groups]);high=np.array([summary[name]['ci95'][1] for name,_,_ in groups])
    ax.bar(x,means,color=['#889b98','#198c87','#c46b35','#777777']);ax.errorbar(x,means,yerr=[np.maximum(0,means-low),np.maximum(0,high-means)],fmt='none',color='black',capsize=4)
    ax.set(xticks=x,xticklabels=[g[0] for g in groups],ylabel='Seconds',title='Held-out survival: mean and 95% CI',ylim=(0,config['seconds']+.5))
    ax=axes[1,1];bottom=np.zeros(4)
    for key,color in [('starved','#d6ac42'),('captured','#d87761'),('timeout','#198c87')]:
        values=np.array([summary[name][key] for name,_,_ in groups]);ax.bar(x,values,bottom=bottom,label=key,color=color);bottom+=values
    ax.set(xticks=x,xticklabels=[g[0] for g in groups],ylabel='Fraction of encounters',title='How first lives ended',ylim=(0,1));ax.legend()
    fig.suptitle(f'Hunger decision-tree pilot | {args.seeds} independent seeds | fixed navigation and predator\nFitness = first-life survival only; confidence intervals resample seeds')
    fig.savefig(out/'outcomes.png',dpi=170);plt.close(fig)
    for seed in range(args.seeds):
        for label,tree in [('Initial',initial[str(seed)]),('Evolved',champions[str(seed)]),('Scripted',BASELINE)]:
            trace=[];episode(tree,90000000+seed*1000,world,config,trace)
            examples[f'Seed {seed+1}: {label}']=dict(tree=tree_text(tree),frames=trace)
    save_replay(out,terrain,examples,config)
    difference=np.array(summary['Evolved']['seed_survival'])-np.array(summary['Initial']['seed_survival'])
    mean,lo,hi=seed_mean_interval(difference[:,None]);summary['paired_survival_change']=dict(mean=float(mean[0]),ci95=[float(lo[0]),float(hi[0])],by_seed=difference.tolist())
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'baseline-tree.txt').write_text(tree_text(BASELINE),encoding='utf-8')
    (out/'design.json').write_text(json.dumps(dict(config=config,budget=vars(args)|{'output':str(out),'terrain_map':str(args.terrain_map) if args.terrain_map else None},features=FEATURES,fitness='Mean first-life survival seconds; no food bonus, distance reward, or capture penalty; no prey respawn.',selection='Two elites, tournament of three; subtree/condition/threshold/action mutation, no crossover; random initial trees, no baseline seeding.',information='Own hunger fraction, position, velocity; all visible food and predator objects with relative/absolute position, distance, route length and waypoint; private recent-visit memory and target/stuck durations. Tree predicates use the listed scalar summaries; action selectors choose from the visible object list.',actions='Approach or move away from a visible object (food or predator, rank 0..2, ordered by distance or route length); move to an unvisited/oldest/random reachable map tile; stay. Missing object => stay; no priority fallback.',fixed='Collision physics, occlusion, known static terrain, pathfinding, waypoint following, away-direction primitive, recent-visit memory mechanics, half-second decision interval. Fixed predator pursues visible prey, follows its last target until reached, then explores its private visited map.',notes=['No co-evolution in this diagnostic.','Controls share the held-out starts; stationary/scripted references are evaluated once, not rerun at every generation.','Survival is capped by the horizon; high timeout fractions imply ceiling effects.','Initial is the generation-0 training-selected champion, not the average random tree.','Three seeds are a feasibility pilot, not conclusive evidence.','Replays show the first held-out start for each seed, not selected highlights.'],wall_seconds=time.perf_counter()-started),indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
