"""Compare predator fitness functions against one fixed scripted prey policy."""
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
from src.paths import ROOT
from src.simulation.hunger_tree import DEFAULTS,BASELINE,PREDATOR_BASELINE,episode,mutate,random_tree,tree_text
from src.simulation.vision_pilot import setup,seed_mean_interval
from src.experiments.hunger_coevolution import predator_score,reproduce
from src.experiments.hunger_tree_pilot import save_replay

WEIGHTS={'capture-only':0.,'progress-0.1':.1,'progress-0.5':.5}
FOOD_ONLY_PREY=deepcopy(BASELINE['no'])
PREY_POLICIES={'avoidance':BASELINE,'food-only':FOOD_ONLY_PREY}
COLORS={'capture-only':'#777777','progress-0.1':'#198c87','progress-0.5':'#c46b35'}


def distance_progress(diagnostics):
    return float(np.clip((diagnostics['initial_distance']-diagnostics['closest_distance'])/max(diagnostics['initial_distance'],1e-12),0,1))


def fitness(result,diagnostics,horizon,weight):
    return predator_score(result,horizon)+weight*distance_progress(diagnostics)


def initial_population(rng,size,depth,initialization):
    if initialization=='random':
        return [random_tree(rng,depth,'predator') for _ in range(size)]
    return [deepcopy(PREDATOR_BASELINE)]+[mutate(PREDATOR_BASELINE,rng,depth,'predator') for _ in range(size-1)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/predator-fitness-pilot')
    p.add_argument('--seeds',type=int,default=3);p.add_argument('--generations',type=int,default=6)
    p.add_argument('--population',type=int,default=6);p.add_argument('--training-starts',type=int,default=4)
    p.add_argument('--test-starts',type=int,default=16);p.add_argument('--eval-every',type=int,default=3)
    p.add_argument('--depth',type=int,default=3)
    p.add_argument('--prey-policy',choices=PREY_POLICIES,default='avoidance',help='Fixed prey: original avoidance or food seeking/exploration only')
    p.add_argument('--initialization',choices=['baseline','random'],default='baseline')
    p.add_argument('--methods',nargs='+',choices=WEIGHTS,default=list(WEIGHTS))
    args=p.parse_args();prey=PREY_POLICIES[args.prey_policy];weights={name:WEIGHTS[name] for name in args.methods}
    if 'capture-only' not in weights:p.error('Include capture-only as the reference fitness condition')
    if min(args.seeds,args.generations,args.training_starts,args.test_starts,args.eval_every,args.depth)<=0 or args.population<4 or args.depth>6:p.error('Positive budgets, population >=4 and depth <=6 required')
    out=args.output;out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();config=dict(DEFAULTS)
    terrain=dict(resolution=96,solid=[0]*96**2,preset='Empty arena');world=setup(terrain)
    (out/'terrain.json').write_text(json.dumps(terrain))
    cache={};requests=0;misses=0;rows=[];training=[]
    models={name:dict(champions={},history={},baseline=PREDATOR_BASELINE) for name in weights}
    def play(tree,scenario):
        nonlocal requests,misses
        requests+=1;key=(json.dumps(tree,sort_keys=True),int(scenario))
        if key not in cache:
            d={};r=episode(prey,int(scenario),world,config,predator_tree=tree,diagnostics=d);cache[key]=(r,d);misses+=1
        return cache[key]
    tests=np.arange(args.test_starts)+350000000
    reference=[play(PREDATOR_BASELINE,i) for i in tests]
    def check(tree,method,seed,generation):
        results=[play(tree,i) for i in tests]
        for i,(r,d) in zip(tests,results):
            captured=r['death']=='captured'
            rows.append(dict(method=method,seed=seed,generation=generation,scenario=int(i),capture=float(captured),capture_time=r['survival'] if captured else '',capped_capture_time=r['survival'] if captured else config['seconds'],progress=distance_progress(d),**r,**d))
        with (out/'measurements.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        return results
    for seed in range(args.seeds):
        rng=np.random.default_rng(41000+seed)
        initial=initial_population(rng,args.population,args.depth,args.initialization)
        (out/f'seed-{seed}-initial-population.json').write_text(json.dumps(initial,indent=2))
        for method,weight in weights.items():
            population=deepcopy(initial);stream=np.random.default_rng(51000+seed)
            for generation in range(args.generations+1):
                scenarios=np.arange(args.training_starts)+3000000+generation*100
                results=[[play(tree,i) for i in scenarios] for tree in population]
                scores=[np.mean([fitness(r,d,config['seconds'],weight) for r,d in batch]) for batch in results]
                champion,children=reproduce(population,scores,stream,'predator',args.depth)
                training.append(dict(method=method,seed=seed,generation=generation,zero_score_fraction=float(np.mean(np.asarray(scores)==0)),no_capture_fraction=float(np.mean([not any(r['death']=='captured' for r,d in batch) for batch in results])),capture_fraction=float(np.mean([r['death']=='captured' for batch in results for r,d in batch])),unique_scores=len(set(np.round(scores,12)))))
                if generation%args.eval_every==0 or generation==args.generations:
                    evaluated=check(champion,method,seed,generation);models[method]['history'][f'{seed}/{generation}']=champion
                    print(f'Check | {method} | seed {seed+1}/{args.seeds} | gen {generation} | captures {sum(r["death"]=="captured" for r,d in evaluated)}/{len(tests)}',flush=True)
                print(f'Train | {method} | seed {seed+1}/{args.seeds} | gen {generation}/{args.generations} | zero scores {training[-1]["zero_score_fraction"]:.0%} | elapsed {time.perf_counter()-started:.1f}s',flush=True)
                population=children
            models[method]['champions'][str(seed)]=champion
            (out/f'seed-{seed}-{method}.txt').write_text(tree_text(champion),encoding='utf-8')
    with (out/'training.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(training[0]));w.writeheader();w.writerows(training)
    gs=sorted({r['generation'] for r in rows});fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    summary={};final_capture={}
    for method in weights:
        for ax,key,title,data in [(axes[0,0],'capture','Actual held-out capture fraction',rows),(axes[0,1],'capped_capture_time','Capture time; failures assigned 30 seconds',rows),(axes[1,0],'closest_distance','Held-out closest distance (lower is closer)',rows),(axes[1,1],'zero_score_fraction','Training population with zero fitness',training)]:
            generations=gs if data is rows else sorted({r['generation'] for r in training})
            values=np.array([[np.mean([r[key] for r in data if r['method']==method and r['seed']==seed and r['generation']==g]) for g in generations] for seed in range(args.seeds)])
            mean,lo,hi=seed_mean_interval(values);ax.plot(generations,mean,label=method,color=COLORS[method]);ax.fill_between(generations,lo,hi,color=COLORS[method],alpha=.15);ax.set(title=title,xlabel='Generation',ylim=(0,1) if key in ('capture','zero_score_fraction') else (0,None))
            if key=='capture':
                final_capture[method]=values[:,-1];diff=values[:,-1]-values[:,0];m,l,h=seed_mean_interval(diff[:,None])
                selected=[r for r in rows if r['method']==method and r['generation']==args.generations]
                cap_times=[r['capture_time'] for r in selected if r['capture']]
                summary[method]=dict(initial_capture=float(values[:,0].mean()),final_capture=float(values[:,-1].mean()),capture_change=float(m[0]),change_ci95=[float(l[0]),float(h[0])],final_seed_captures=values[:,-1].tolist(),mean_capped_capture_time=float(np.mean([r['capped_capture_time'] for r in selected])),capture_time_when_captured=float(np.mean(cap_times)) if cap_times else None)
    axes[0,0].axhline(np.mean([r['death']=='captured' for r,d in reference]),ls='--',color='#444444',label='Original scripted predator')
    for ax in axes.flat:ax.legend(fontsize=8)
    fig.suptitle(f'Predator-only fitness diagnostic | {args.seeds} seeds | fixed {args.prey_policy} prey | {args.initialization} initialization\nMean and pointwise 95% CIs across seeds; speeds/vision/hunger unchanged')
    fig.savefig(out/'outcomes.png',dpi=170);plt.close(fig)
    for method in [name for name in weights if name!='capture-only']:
        m,l,h=seed_mean_interval((final_capture[method]-final_capture['capture-only'])[:,None]);summary[method]['final_capture_minus_capture_only']=dict(mean=float(m[0]),ci95=[float(l[0]),float(h[0])])
    summary['scripted_reference_capture']=float(np.mean([r['death']=='captured' for r,d in reference]))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    default=models['capture-only'];prey_champs={str(seed):prey for seed in range(args.seeds)};prey_history={key:prey for key in default['history']}
    (out/'trees.json').write_text(json.dumps(dict(champions=prey_champs,history=prey_history,predator_champions=default['champions'],predator_history=default['history'],predator_conditions=models,baseline_predator=PREDATOR_BASELINE),indent=2))
    examples={}
    for seed in range(args.seeds):
        for method in weights:
            pred=models[method]['champions'][str(seed)];trace=[];episode(prey,int(tests[0]),world,config,trace,predator_tree=pred)
            examples[f'Seed {seed+1}: {method}']=dict(tree='PREDATOR\n'+tree_text(pred)+'\n\nFIXED PREY\n'+tree_text(prey),frames=trace)
    save_replay(out,terrain,examples,config)
    (out/'design.json').write_text(json.dumps(dict(config=config,budget=vars(args)|{'output':str(out)},evaluation_seed_base=350000000,fitness={'capture_only':'capture*(2-time/horizon)','shaped':'capture_only + weight*clip((initial_distance-closest_distance)/initial_distance,0,1)','weights':weights},notes=[f'Initialization: {args.initialization}; baseline is seeded only in baseline mode. Random mode samples independent predator trees; the script remains a separate reference.', 'Generation 0 is the training-selected winner of the initial population, not its average.', f'One fixed {args.prey_policy} prey shared by every seed and condition; no prey evolution.','Identical initial predator population per seed, mutation RNG seed and training scenario schedule across conditions.','Training starts resample each generation; held-out suite is common across all seeds and conditions. Confidence intervals vary evolutionary seeds conditional on this suite.','Closest distance is sampled at 0.1-second physics boundaries and exact death positions. Global separation is used only for fitness, not controller observations.','Initial food, physiology, speeds, vision, navigation and tree primitives unchanged.','The bounded shaping bonus can produce reward without capture: actual capture rate is the primary evaluation.','Duplicate episodes and the scripted reference are cached. References are not evolved or rerun unnecessarily.','Small pilot; no claim of an optimal weight or learning benefit without improved held-out captures.'],episodes_simulated=misses,evaluation_requests=requests,wall_seconds=time.perf_counter()-started),indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
