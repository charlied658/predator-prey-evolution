"""Evaluate archived pursuit controllers against stable and historical opponents."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vision_pilot import episodes,setup,GENES


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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path);args=parser.parse_args();out=args.run/'progress-diagnostic';out.mkdir(exist_ok=False)
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
