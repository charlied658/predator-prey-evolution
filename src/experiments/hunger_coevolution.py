"""Paired small predator/prey tree co-evolution and fixed-predator control."""
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
from src.simulation.hunger_tree import DEFAULTS,BASELINE,FEATURES,PREDATOR_FEATURES,PREDATOR_BASELINE,episode,mutate,random_tree,tree_text
from src.simulation.vision_pilot import setup,seed_mean_interval
from src.experiments.hunger_tree_pilot import save_replay


def predator_score(result,horizon):
    return (2-result['survival']/horizon) if result['death']=='captured' else 0.


def reproduce(population,scores,rng,role,depth):
    order=np.lexsort((rng.random(len(scores)),-np.asarray(scores)))
    champion=deepcopy(population[int(order[0])]);children=[deepcopy(population[int(i)]) for i in order[:2]]
    while len(children)<len(population):
        candidates=rng.choice(len(population),3,replace=False)
        winner=int(candidates[np.argmax(np.asarray(scores)[candidates])])
        children.append(mutate(population[winner],rng,depth,role))
    return champion,children


def summarize(rows,out,seeds,generations,horizon):
    def values(condition,evaluation,key):
        if not any(r['condition']==condition and r['evaluation']==evaluation for r in rows):return np.full((seeds,len(generations)),np.nan)
        return np.array([[np.mean([r[key] for r in rows if r['condition']==condition and r['evaluation']==evaluation and r['seed']==seed and r['generation']==g]) for g in generations] for seed in range(seeds)])
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    def curve(ax,data,label,color):
        mean,lo,hi=seed_mean_interval(data);ax.plot(generations,mean,marker='o',label=label,color=color);ax.fill_between(generations,lo,hi,color=color,alpha=.17)
    for condition,label,color in [('coevolution','Co-evolved prey','#198c87'),('fixed-control','Fixed-predator-trained prey','#c46b35')]:
        if any(r['condition']==condition for r in rows):curve(axes[0,0],values(condition,'fixed_predator','survival'),label,color)
    axes[0,0].set(title='Prey vs the SAME fixed baseline predator',ylabel='Mean first-life survival (s)',ylim=(0,horizon+.5))
    curve(axes[0,1],values('coevolution','fixed_prey','captured'),'Evolving predator vs fixed reference prey','#c66c58')
    axes[0,1].set(title='Predator vs the SAME fixed reference prey',ylabel='Capture fraction',ylim=(0,1))
    curve(axes[1,0],values('coevolution','current_pair','survival'),'Current predator / current prey','#75669f')
    axes[1,0].set(title='Contemporary matchup (not absolute progress)',ylabel='Prey survival (s)',ylim=(0,horizon+.5))
    heat=np.array([[np.mean([r['captured'] for r in rows if r['evaluation']=='historical' and r['predator_generation']==a and r['prey_generation']==b]) for b in generations] for a in generations])
    ax=axes[1,1];im=ax.imshow(heat,vmin=0,vmax=1,origin='lower',cmap='magma')
    ax.set(title='Historical matchups: capture fraction',xticks=range(len(generations)),xticklabels=generations,yticks=range(len(generations)),yticklabels=generations,xlabel='Prey generation',ylabel='Predator generation')
    for i in range(len(generations)):
        for j in range(len(generations)):ax.text(j,i,f'{heat[i,j]:.0%}',ha='center',va='center',color='white' if heat[i,j]<.65 else 'black')
    fig.colorbar(im,ax=ax)
    for ax in (axes[0,0],axes[0,1],axes[1,0]):ax.set_xlabel('Generation');ax.legend(fontsize=8)
    fig.suptitle(f'Hunger tree co-evolution | {seeds} seeds | empty arena, unchanged physiology/speeds\nMean and pointwise 95% bootstrap intervals across seeds')
    fig.savefig(out/'outcomes.png',dpi=170);plt.close(fig)
    summary={}
    for name,data in [('prey_coevolved_vs_baseline',values('coevolution','fixed_predator','survival')),('prey_fixed_control_vs_baseline',values('fixed-control','fixed_predator','survival')),('predator_vs_fixed_source_prey',values('coevolution','fixed_prey','captured')),('contemporary_prey_survival',values('coevolution','current_pair','survival'))]:
        if np.isnan(data).all():continue
        change=data[:,-1]-data[:,0];mean,lo,hi=seed_mean_interval(change[:,None])
        summary[name]=dict(initial=float(data[:,0].mean()),final=float(data[:,-1].mean()),change=float(mean[0]),change_ci95=[float(lo[0]),float(hi[0])],seed_changes=change.tolist())
    if any(r['condition']=='fixed-control' for r in rows):
        difference=values('coevolution','fixed_predator','survival')[:,-1]-values('fixed-control','fixed_predator','survival')[:,-1]
        mean,lo,hi=seed_mean_interval(difference[:,None]);summary['final_prey_coevolution_minus_fixed']=dict(mean=float(mean[0]),ci95=[float(lo[0]),float(hi[0])],by_seed=difference.tolist())
    (out/'summary.json').write_text(json.dumps(summary,indent=2));return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'results/hunger-tree-long-01')
    p.add_argument('--output',type=Path,default=ROOT/'results/hunger-coevolution-pilot')
    p.add_argument('--seeds',type=int,default=3);p.add_argument('--generations',type=int,default=6)
    p.add_argument('--population',type=int,default=6);p.add_argument('--opponents',type=int,default=2)
    p.add_argument('--training-starts',type=int,default=2);p.add_argument('--test-starts',type=int,default=6)
    p.add_argument('--eval-every',type=int,default=3);p.add_argument('--depth',type=int,default=3)
    p.add_argument('--initialization',choices=['source','random'],default='source')
    p.add_argument('--no-fixed-control',action='store_true',help='Skip the separate fixed-predator prey training population')
    args=p.parse_args()
    if min(args.seeds,args.generations,args.opponents,args.training_starts,args.test_starts,args.eval_every,args.depth)<=0 or args.population<4 or args.opponents>args.population or args.depth>6:p.error('Positive budgets; population >=4, opponents <= population, depth <=6')
    if args.initialization=='source':
        source=json.loads((args.source/'trees.json').read_text());source_design=json.loads((args.source/'design.json').read_text())
        source_keys=sorted(source['champions'],key=int)
        if args.seeds>len(source_keys):p.error('Source must contain at least one different prey controller per pilot seed')
        config=dict(DEFAULTS,**source_design['config'])
    else:
        config=dict(DEFAULTS)
    if not config['predator']:p.error('Predator must be enabled')
    out=args.output;out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    terrain=dict(resolution=96,solid=[0]*96**2,preset='Empty arena');world=setup(terrain)
    (out/'terrain.json').write_text(json.dumps(terrain))
    rows=[];cache={};misses=0;requests=0
    saved=dict(champions={},history={},predator_champions={},predator_history={},fixed_champions={},fixed_history={},source_preys={},baseline_predator=PREDATOR_BASELINE)
    def play(prey,predator,scenario):
        nonlocal misses,requests
        requests+=1;key=(json.dumps(prey,sort_keys=True),json.dumps(predator,sort_keys=True),int(scenario))
        if key not in cache:
            cache[key]=episode(prey,int(scenario),world,config,predator_tree=predator);misses+=1
        return cache[key]
    def check(prey,predator,ids,seed,generation,condition,evaluation,pg=-1,qg=-1):
        for scenario in ids:
            result=play(prey,predator,scenario)
            rows.append(dict(condition=condition,evaluation=evaluation,seed=seed,generation=generation,predator_generation=pg,prey_generation=qg,scenario=int(scenario),captured=float(result['death']=='captured'),predator_score=predator_score(result,config['seconds']),**result))
        with (out/'measurements.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for seed in range(args.seeds):
        anchor=deepcopy(source['champions'][source_keys[seed]] if args.initialization=='source' else BASELINE['no']);saved['source_preys'][str(seed)]=anchor
        init=np.random.default_rng(10100+seed)
        population=([random_tree(init,args.depth,'prey') for _ in range(args.population)] if args.initialization=='random' else [deepcopy(anchor)]+[mutate(anchor,init,args.depth) for _ in range(args.population-1)])
        prey_pop=deepcopy(population);fixed_pop=deepcopy(population)
        pred_pop=([random_tree(init,args.depth,'predator') for _ in range(args.population)] if args.initialization=='random' else [deepcopy(PREDATOR_BASELINE)]+[mutate(PREDATOR_BASELINE,init,args.depth,'predator') for _ in range(args.population-1)])
        (out/f'seed-{seed}-initial-populations.json').write_text(json.dumps(dict(prey=prey_pop,predator=pred_pop),indent=2))
        streams=[np.random.default_rng(20100+seed*3+i) for i in range(3)];sampling=np.random.default_rng(30100+seed)
        tests=np.arange(args.test_starts)+200000000+seed*1000
        for generation in range(args.generations+1):
            ids=np.arange(args.training_starts)+2000000+seed*10000+generation*100
            shared_pred=[0]+sampling.choice(np.arange(1,args.population),args.opponents-1,replace=False).tolist()
            shared_prey=[0]+sampling.choice(np.arange(1,args.population),args.opponents-1,replace=False).tolist()
            pairs=sorted({(a,b) for a in shared_pred for b in range(args.population)}|{(a,b) for a in range(args.population) for b in shared_prey})
            measured={(a,b):[play(prey_pop[b],pred_pop[a],i) for i in ids] for a,b in pairs}
            prey_scores=[np.mean([r['survival'] for a in shared_pred for r in measured[(a,b)]]) for b in range(args.population)]
            pred_scores=[np.mean([predator_score(r,config['seconds']) for b in shared_prey for r in measured[(a,b)]]) for a in range(args.population)]
            fixed_scores=[] if args.no_fixed_control else [np.mean([play(tree,PREDATOR_BASELINE,i)['survival'] for i in ids]) for tree in fixed_pop]
            prey_champ,prey_next=reproduce(prey_pop,prey_scores,streams[0],'prey',args.depth)
            pred_champ,pred_next=reproduce(pred_pop,pred_scores,streams[1],'predator',args.depth)
            if not args.no_fixed_control:fixed_champ,fixed_next=reproduce(fixed_pop,fixed_scores,streams[2],'prey',args.depth)
            else:fixed_champ=None;fixed_next=[]
            if generation%args.eval_every==0 or generation==args.generations:
                key=f'{seed}/{generation}';saved['history'][key]=prey_champ;saved['predator_history'][key]=pred_champ
                if fixed_champ is not None:saved['fixed_history'][key]=fixed_champ
                check(prey_champ,PREDATOR_BASELINE,tests,seed,generation,'coevolution','fixed_predator')
                if fixed_champ is not None:check(fixed_champ,PREDATOR_BASELINE,tests,seed,generation,'fixed-control','fixed_predator')
                check(anchor,pred_champ,tests,seed,generation,'coevolution','fixed_prey')
                check(prey_champ,pred_champ,tests,seed,generation,'coevolution','current_pair',generation,generation)
            print(f'Coevolution | seed {seed+1}/{args.seeds} | generation {generation}/{args.generations} | prey train {max(prey_scores):.2f}s | predator score {max(pred_scores):.3f} | elapsed {time.perf_counter()-started:.1f}s',flush=True)
            prey_pop,pred_pop,fixed_pop=prey_next,pred_next,fixed_next
        saved['champions'][str(seed)]=prey_champ;saved['predator_champions'][str(seed)]=pred_champ
        if fixed_champ is not None:saved['fixed_champions'][str(seed)]=fixed_champ
        for role,tree in [('prey',prey_champ),('predator',pred_champ),('fixed-control-prey',fixed_champ)]:
            if tree is not None:(out/f'seed-{seed}-{role}.txt').write_text(tree_text(tree),encoding='utf-8')
        (out/'trees.json').write_text(json.dumps(saved,indent=2))
    generations=sorted({int(key.split('/')[1]) for key in saved['history']})
    for seed in range(args.seeds):
        tests=np.arange(args.test_starts)+200000000+seed*1000
        for a in generations:
            for b in generations:
                check(saved['history'][f'{seed}/{b}'],saved['predator_history'][f'{seed}/{a}'],tests,seed,args.generations,'coevolution','historical',a,b)
        print(f'Historical matchups | seed {seed+1}/{args.seeds} | elapsed {time.perf_counter()-started:.1f}s',flush=True)
    summary=summarize(rows,out,args.seeds,generations,config['seconds'])
    examples={}
    for seed in range(args.seeds):
        for g in (0,args.generations):
            key=f'{seed}/{g}';trace=[];episode(saved['history'][key],200000000+seed*1000,world,config,trace,predator_tree=saved['predator_history'][key])
            examples[f'Seed {seed+1}, generation {g}']=dict(tree='PREDATOR\n'+tree_text(saved['predator_history'][key])+'\n\nPREY\n'+tree_text(saved['history'][key]),frames=trace)
    save_replay(out,terrain,examples,config)
    design=dict(config=config,budget=vars(args)|{'source':str(args.source.resolve()) if args.initialization=='source' else None,'output':str(out)},predator_features=PREDATOR_FEATURES,prey_features=FEATURES,fitness={'prey':'First-life survival seconds only','predator':'If captured: 2-survival/horizon. Starvation or timeout: zero.'},
                notes=[f'Initialization: {args.initialization}. Random mode uses independent random trees for both species, without baseline seeding or inherited controllers; fixed predator and food-only prey are evaluation references only.', f'Separate fixed-predator control enabled: {not args.no_fixed_control}. Generation 0 is the training-selected winner, not a random-population average.','Empty arena; random mode uses default physiology, speeds, vision and navigation; source mode inherits configuration.','Source mode: first N source seed controllers plus mutated variants. If enabled, the fixed control shares the initial prey population.','Source mode predators start with the scripted baseline and mutants; random mode predators are independently generated.','Shared current opponents include the previous elite and sampled others. Both populations update simultaneously.','Fixed control uses identical training starts and prey population/generation budgets. Coevolution also trains predators and tests multiple opponents: total compute is not matched.','Fresh held-out starts (200000000+1000*seed) are disjoint from previous pilot tests and all training starts.','Duplicate episodes are cached, including historical diagonals. Test evaluations never select or mutate controllers.','Historical matrices are aggregated over seeds, not independent cross-seed opponents.','Few seeds/generations can establish feasibility but not a reliable arms race.'],episodes_simulated=misses,evaluation_requests=requests,wall_seconds=time.perf_counter()-started)
    (out/'design.json').write_text(json.dumps(design,indent=2));print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
