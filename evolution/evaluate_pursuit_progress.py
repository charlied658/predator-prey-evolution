"""Evaluate archived pursuit controllers against stable and historical opponents."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vision_pilot import episodes,setup,GENES,seed_mean_interval


def plot_capture_summary(rows,conditions,generations,seeds,out):
    fig,axes=plt.subplots(1,len(conditions),figsize=(11,3.6),layout='constrained',squeeze=False)
    for ax,name in zip(axes[0],conditions):
        for key,label,color in [('current_capture','Current prey','#ba6954'),('panel_predator_capture','Fixed prey panel','#188b87')]:
            values=[]
            for seed in range(seeds):
                sub=sorted([r for r in rows if r['condition']==name and int(r['seed'])==seed],key=lambda r:int(r['generation']))
                ys=[float(r[key]) for r in sub];values.append(ys);ax.plot(generations,ys,color=color,alpha=.17,lw=1)
            ax.plot(generations,np.mean(values,axis=0),color=color,lw=2.5,label=label)
        ax.set(title=name,xlabel='Generation',ylabel='Predator capture fraction',ylim=(-.02,1.02));ax.legend(fontsize=8)
    fig.suptitle('Same predators, two ways to measure progress\nThick lines: 3-seed averages; faint lines: individual seeds')
    fig.savefig(out/'capture-summary.png',dpi=170);plt.close(fig)


def compare_training(run,baseline,opponent_comparison=False,benchmark_reference=None):
    """Compare selection batch sizes using one shared frozen opponent panel."""
    out=run/('opponent-comparison' if opponent_comparison else 'selection-comparison');out.mkdir(exist_ok=False);started=time.perf_counter()
    old_label='Co-evolution' if opponent_comparison else '2 encounters'
    new_label='Fixed opponent' if opponent_comparison else '8 encounters'
    paths={old_label:baseline,new_label:run}
    designs={k:json.loads((v/'design.json').read_text()) for k,v in paths.items()}
    histories={k:json.loads((v/'history.json').read_text()) for k,v in paths.items()}
    d=designs[new_label];seeds=d['seeds'];world=setup(json.loads((run/'terrain.json').read_text()));ranges=d['conditions']['equal'];gs=list(range(0,d['generations']+1,d['evaluation_interval']));rows=[];summary={}
    for key in ['genes','population','food_encoding','speed','dt','steps','food_reward','training_mode']:
        assert designs[old_label][key]==d[key],f'Confounded setting: {key}'
    np.testing.assert_array_equal(json.loads((baseline/'terrain.json').read_text())['solid'],json.loads((run/'terrain.json').read_text())['solid'])
    if opponent_comparison:
        assert designs[old_label]['training_starts']==d['training_starts']
        assert designs[old_label].get('opponent_mode','coevolution')=='coevolution' and d['opponent_mode']=='fixed'
    else:
        assert designs[old_label]['training_starts']==2 and d['training_starts']==8
    reference_panel=None
    panel_description='Other baseline seeds, initial and final controllers'
    if benchmark_reference:
        reference_panel=[]
        current_ids=set(range(d.get('seed_offset',0),d.get('seed_offset',0)+seeds))
        for mode in ('coevolution','fixed'):
            path=benchmark_reference/mode
            ref=json.loads((path/'design.json').read_text());archive=json.loads((path/'history.json').read_text())
            reference_ids=set(range(ref.get('seed_offset',0),ref.get('seed_offset',0)+ref['seeds']))
            if current_ids & reference_ids:raise ValueError('Reference benchmark must use separate evolutionary seeds; set --seed-offset')
            for key in ('genes','food_encoding','speed','dt','steps','food_reward'):
                assert ref[key]==d[key],f'Incompatible benchmark: {key}'
            np.testing.assert_array_equal(json.loads((path/'terrain.json').read_text())['solid'],json.loads((run/'terrain.json').read_text())['solid'])
            reference_panel += [np.array(archive[f'equal/{seed}/{ref["generations"]}']) for seed in range(ref['seeds'])]
        panel_description=f'Frozen final controllers from BOTH treatments in {benchmark_reference}; independent evolutionary seeds'
    elif opponent_comparison:
        panel_description='Other seeds: initial controllers once, plus final controllers from BOTH treatments; shared between treatments'
    def champ(mode,seed,g):return np.array(histories[mode][f'equal/{seed}/{g}'])
    def evaluate(pairs,ids):
        a=np.repeat(np.array([x for x,y in pairs]),len(ids),axis=0);b=np.repeat(np.array([y for x,y in pairs]),len(ids),axis=0)
        return episodes(a,b,np.tile(ids,len(pairs)),ranges,world,food_encoding=d['food_encoding']).reshape(len(pairs),len(ids),7)
    for seed in range(seeds):
        np.testing.assert_array_equal(champ(old_label,seed,0),champ(new_label,seed,0))
        ids=np.arange(24)+80000+seed*100
        if reference_panel is not None:
            panel=reference_panel
        elif opponent_comparison:
            panel=[controller for other in range(seeds) if other!=seed for controller in (champ(old_label,other,0),champ(old_label,other,designs[old_label]['generations']),champ(new_label,other,d['generations']))]
        else:
            panel=[champ(old_label,other,g) for other in range(seeds) if other!=seed for g in (0,designs[old_label]['generations'])]
        for mode in paths:
            prey_pairs=[(opponent[0],champ(mode,seed,g)[1]) for g in gs for opponent in panel]
            predator_pairs=[(champ(mode,seed,g)[0],opponent[1]) for g in gs for opponent in panel]
            prey=evaluate(prey_pairs,ids).reshape(len(gs),len(panel),len(ids),7)
            predator=evaluate(predator_pairs,ids).reshape(len(gs),len(panel),len(ids),7)
            for i,g in enumerate(gs):rows.append(dict(training=mode,seed=seed,generation=g,predator_capture=float(predator[i,:,:,2].mean()),predator_fitness=float(predator[i,:,:,0].mean()),prey_survival=float(1-prey[i,:,:,2].mean()),food=float(prey[i,:,:,6].mean()),prey_fitness=float(prey[i,:,:,1].mean())))
            with (out/'measurements.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            print(f'{mode} | seed {seed+1}/{seeds} | benchmark complete | {time.perf_counter()-started:.1f}s',flush=True)
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,key,title in zip(axes.flat,['predator_capture','prey_survival','food','prey_fitness'],['Predator capture','Prey survival','Prey pellets collected','Prey total fitness']):
        for mode,color in [(old_label,'#ba6954'),(new_label,'#188b87')]:
            values=[]
            for seed in range(seeds):
                sub=sorted([r for r in rows if r['training']==mode and r['seed']==seed],key=lambda r:r['generation']);ys=[r[key] for r in sub];values.append(ys)
            values=np.array(values);mean,low,high=seed_mean_interval(values);ax.plot(gs,mean,color=color,lw=2.5,label=mode);ax.fill_between(gs,low,high,color=color,alpha=.18)
            summary.setdefault(mode,{})[key]=dict(initial=float(values[:,0].mean()),final=float(values[:,-1].mean()),final_seed_means=values[:,-1].tolist(),final_mean_ci95=[float(low[-1]),float(high[-1])],mean_step_change=float(np.abs(np.diff(values,axis=1)).mean()))
        ax.set(title=title,xlabel='Generation',ylabel='Pellets' if key=='food' else 'Fitness' if key=='prey_fitness' else 'Fraction');ax.legend(fontsize=9)
        if key in ('predator_capture','prey_survival'):ax.set_ylim(0,1)
        elif key=='food':ax.set_ylim(0,None)
    fig.suptitle(('Fixed opponent versus co-evolution' if opponent_comparison else 'Does judging candidates on more encounters improve evolution?')+f'\nMean and pointwise 95% bootstrap CI across {seeds} independent seeds')
    fig.savefig(out/('opponent-comparison.png' if opponent_comparison else 'selection-comparison.png'),dpi=170);plt.close(fig)
    difference={}
    diff_fig,diff_axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,key,title in zip(diff_axes.flat,['predator_capture','prey_survival','food','prey_fitness'],['Predator capture','Prey survival','Pellets collected','Prey total fitness']):
        def seed_values(mode):
            return np.array([[r[key] for r in sorted([r for r in rows if r['training']==mode and r['seed']==seed],key=lambda r:r['generation'])] for seed in range(seeds)])
        delta=seed_values(old_label)-seed_values(new_label)
        mean,low,high=seed_mean_interval(delta)
        ax.axhline(0,color='#777777',lw=1,ls='--');ax.plot(gs,mean,color='#5368a0',lw=2);ax.fill_between(gs,low,high,color='#5368a0',alpha=.2)
        ax.set(title=title,xlabel='Generation',ylabel=f'{old_label} minus {new_label}')
        difference[key]=dict(final_mean=float(mean[-1]),final_ci95=[float(low[-1]),float(high[-1])],final_seed_differences=delta[:,-1].tolist())
    diff_fig.suptitle('Paired treatment differences: positive favours '+old_label+'\nMean and pointwise 95% bootstrap CI across matched seed pairs')
    diff_fig.savefig(out/'paired-differences.png',dpi=170);plt.close(diff_fig)
    summary['paired_differences']=difference
    (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'design.json').write_text(json.dumps(dict(runs={k:str(v) for k,v in paths.items()},panel=panel_description,confidence_intervals='Pointwise 95% percentile bootstrap of seed trajectories, 10000 resamples; paired differences preserve matched seed pairs',test_starts='24 per seed, IDs 80000 + 100*seed + [0..23]; not used in selection or prior plots',training_difference=('Same candidate generations and 8 starts; frozen generation-zero opponent versus contemporary opponent population; deterministic fixed repeats omitted, so compute is not equal' if opponent_comparison else '2 vs 8 fresh encounters; 8 includes the same first 2; 4x training evaluations at equal generations'),notes=['Search opportunities are matched by generations and candidates; actual simulator cost is reported by each source run.',f'{seeds} evolutionary seeds; individual starting arrangements are not independent evolutionary replicates. Confidence intervals are conditional on the frozen panel; bands are not simultaneous across generations.','No smoothing or best-checkpoint selection.'],seconds=time.perf_counter()-started),indent=2));print(json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path);parser.add_argument('--compare-training',type=Path);parser.add_argument('--compare-opponents',type=Path);parser.add_argument('--benchmark-reference',type=Path);args=parser.parse_args()
    if args.compare_opponents:
        compare_training(args.run,args.compare_opponents,opponent_comparison=True,benchmark_reference=args.benchmark_reference)
        return
    if args.compare_training:
        compare_training(args.run,args.compare_training)
        return
    out=args.run/'progress-diagnostic';out.mkdir(exist_ok=False)
    started=time.perf_counter();design=json.loads((args.run/'design.json').read_text());history=json.loads((args.run/'history.json').read_text());world=setup(json.loads((args.run/'terrain.json').read_text()))
    seeds=design['seeds'];conditions=design['conditions'];encoding=design['food_encoding'];generations=sorted({int(k.split('/')[-1]) for k in history});historic=generations[::2]
    if generations[-1] not in historic:historic.append(generations[-1])
    original=list(csv.DictReader((args.run/'measurements.csv').open()));rows=[];matrices={};summary={}
    def champ(name,seed,g):return np.array(history[f'{name}/{seed}/{g}'])
    def evaluate(pairs,ids,ranges,capture=True):
        a=np.repeat(np.array([x for x,y in pairs]),len(ids),axis=0);b=np.repeat(np.array([y for x,y in pairs]),len(ids),axis=0)
        return episodes(a,b,np.tile(ids,len(pairs)),ranges,world,capture=capture,food_encoding=encoding).reshape(len(pairs),len(ids),7)
    for name,ranges in conditions.items():
        for seed in range(seeds):
            ids=np.arange(design['test_starts'])+5000+seed*100
            # Other independently evolving seeds supply a stable panel with
            # initial and final strategies. This is retrospective evaluation;
            # panel scores never affect selection.
            panel=[champ(name,other,g) for other in range(seeds) if other!=seed for g in (generations[0],generations[-1])]
            if not panel:raise ValueError('Stable panel requires at least two evolutionary seeds')
            current=evaluate([tuple(champ(name,seed,g)) for g in generations],ids,ranges)
            for i,g in enumerate(generations):
                expected=[float(r['capture']) for r in original if r['condition']==name and int(r['seed'])==seed and int(r['generation'])==g]
                np.testing.assert_array_equal(current[i,:,2],expected)
            prey_pairs=[(opponent[0],champ(name,seed,g)[1]) for g in generations for opponent in panel]
            predator_pairs=[(champ(name,seed,g)[0],opponent[1]) for g in generations for opponent in panel]
            prey=evaluate(prey_pairs,ids,ranges).reshape(len(generations),len(panel),len(ids),7)
            predator=evaluate(predator_pairs,ids,ranges).reshape(len(generations),len(panel),len(ids),7)
            solo=evaluate([(np.zeros(GENES),champ(name,seed,g)[1]) for g in generations],ids,ranges,capture=False)
            for i,g in enumerate(generations):
                rows.append(dict(condition=name,seed=seed,generation=g,current_capture=float(current[i,:,2].mean()),current_food=float(current[i,:,6].mean()),panel_predator_capture=float(predator[i,:,:,2].mean()),panel_prey_survival=float(1-prey[i,:,:,2].mean()),panel_food=float(prey[i,:,:,6].mean()),solo_food=float(solo[i,:,6].mean())))
            matrix=evaluate([(champ(name,seed,a)[0],champ(name,seed,b)[1]) for a in historic for b in historic],ids,ranges)[:,:,2].mean(axis=1).reshape(len(historic),len(historic))
            matrices[f'{name}/{seed}']=matrix.tolist()
            with (out/'measurements.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            print(f'{name} | seed {seed+1}/{seeds} | evaluation complete | {time.perf_counter()-started:.1f}s',flush=True)
    fig,axes=plt.subplots(len(conditions),3,figsize=(13,3.3*len(conditions)),layout='constrained',squeeze=False)
    for row_index,name in enumerate(conditions):
        settings=[('Predator capture',[('current_capture','Current opponent','#ba6954'),('panel_predator_capture','Fixed panel','#188b87')]),('Prey survival',[('current_survival','Current opponent','#ba6954'),('panel_prey_survival','Fixed panel','#188b87')]),('Prey food collection',[('current_food','Current opponent','#ba6954'),('panel_food','Fixed panel','#188b87'),('solo_food','No predator','#9270ab')])]
        group=[r for r in rows if r['condition']==name]
        for ax,(title,metrics) in zip(axes[row_index],settings):
            for key,label,color in metrics:
                ys=[]
                for seed in range(seeds):
                    sub=sorted([r for r in group if r['seed']==seed],key=lambda r:r['generation'])
                    values=np.array([1-r['current_capture'] if key=='current_survival' else r[key] for r in sub]);ys.append(values);ax.plot(generations,values,color=color,alpha=.17,lw=1)
                ax.plot(generations,np.mean(ys,axis=0),color=color,lw=2.2,label=label)
            ax.set(title=f'{name}: {title}',xlabel='Generation',ylabel='Pellets' if 'food' in title else 'Fraction',ylim=(0,None));ax.legend(fontsize=8)
            if 'food' not in title:ax.set_ylim(-.02,1.02)
        summary[name]={}
        for key in ['current_capture','panel_predator_capture','panel_prey_survival','current_food','panel_food','solo_food']:
            values=np.array([[r[key] for r in sorted([r for r in group if r['seed']==seed],key=lambda r:r['generation'])] for seed in range(seeds)])
            summary[name][key]=dict(initial=float(values[:,0].mean()),final=float(values[:,-1].mean()),mean_step_change=float(np.abs(np.diff(values,axis=1)).mean()))
    fig.suptitle('Does opponent change explain the jagged progress curves?\nThick lines: seed means; faint lines: individual seeds. Fixed panel is identical at every checkpoint.',fontsize=13)
    fig.savefig(out/'progress-comparison.png',dpi=160);plt.close(fig)
    plot_capture_summary(rows,conditions,generations,seeds,out)
    fig,axes=plt.subplots(1,len(conditions),figsize=(12,4),layout='constrained',squeeze=False)
    for ax,name in zip(axes[0],conditions):
        matrix=np.mean([matrices[f'{name}/{seed}'] for seed in range(seeds)],axis=0)
        im=ax.imshow(matrix,vmin=0,vmax=1,cmap='magma',origin='lower')
        for i in range(len(historic)):
            for j in range(len(historic)):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if matrix[i,j]<.6 else 'black')
        ax.set(title=name,xlabel='Prey generation',ylabel='Predator generation',xticks=range(len(historic)),xticklabels=historic,yticks=range(len(historic)),yticklabels=historic)
        summary[name]['historical']=dict(early_vs_early=float(matrix[0,0]),late_vs_early=float(matrix[-1,0]),early_vs_late=float(matrix[0,-1]),late_vs_late=float(matrix[-1,-1]))
    fig.colorbar(im,ax=axes[0].tolist(),label='Capture fraction');fig.suptitle('Historical matchups: average across three evolutionary seeds');fig.savefig(out/'historical-matchups.png',dpi=170);plt.close(fig)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'historical-matrices.json').write_text(json.dumps(dict(generations=historic,matrices=matrices)))
    (out/'design.json').write_text(json.dumps(dict(source=str(args.run),panel='Initial and final controllers from the other evolutionary seeds, fixed across evaluated generations; retrospective only',held_out_starts=design['test_starts'],generations=generations,historical_generations=historic,training_unchanged=True,notes=['No requirement that fitness increase monotonically.','Solo foraging masks opponent information as well as disabling capture.','Historical matrix alone does not prove cycling; dense temporal sampling and replication would be needed.'],seconds=time.perf_counter()-started),indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
