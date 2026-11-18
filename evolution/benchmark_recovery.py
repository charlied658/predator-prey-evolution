"""Paired recovery-rule benchmark on saved controllers; no retraining."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from single_pellet import evaluate,hybrid_exploration_sites,HYBRID_INSTRUCTIONS
from vision_pilot import DT,setup,seed_mean_interval


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--starts',type=int,default=32);p.add_argument('--seconds',type=float,default=120);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--output',type=Path,default=Path('results/recovery-benchmark'))
    args=p.parse_args()
    if min(args.starts,args.seconds,args.batch_size)<=0 or args.starts>10000:p.error('Use positive values; starts must not exceed 10000')
    if abs(round(args.seconds/DT)*DT-args.seconds)>1e-8:p.error('Seconds must be a multiple of the simulation timestep')
    design=json.loads((args.run/'design.json').read_text());saved=json.loads((args.run/'instructions.json').read_text())
    for key in ('steering_limit_degrees','minimum_throttle','goal_reached_distance'):
        if saved[key]!=HYBRID_INSTRUCTIONS[key]:raise ValueError(f'Saved controller rule mismatch: {key}')
    count=design['pellets'];grid=design['grid_cells'];champion_path=args.run/f'{count}-pellets'/'champions.json';champions=np.array(json.loads(champion_path.read_text()))
    if len(champions)<2:raise ValueError('At least two independent saved controllers are needed for seed-based CIs')
    args.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    terrain=json.loads((args.run/'terrain.json').read_text());world=setup(terrain);_,eligible=hybrid_exploration_sites(world,grid);accessible=int(eligible.sum())
    times=np.unique(np.r_[np.arange(0,args.seconds,1.),args.seconds]);n=len(champions);cases=[];curves=[];raw={}
    labels=[('Recovery off',False),('Recovery on',True)]
    for seed,genome in enumerate(champions):
        ids=np.arange(args.starts)+92000000+seed*10000
        for label,enabled in labels:
            collected={key:[] for key in ('food','visited','reachable_visited')}
            for offset in range(0,args.starts,args.batch_size):
                batch=ids[offset:offset+args.batch_size];progress=[]
                result=evaluate(np.repeat(genome[None],len(batch),axis=0),batch,world,pellet_count=count,food_layout='scattered',grid_cells=grid,controller='hybrid',duration_seconds=args.seconds,recovery=enabled,progress=progress)
                sampled_times=np.array([item['seconds'] for item in progress])
                batch_curves={}
                for key in collected:
                    measured=np.stack([item[key] for item in progress])
                    batch_curves[key]=np.array([np.interp(times,sampled_times,measured[:,j]) for j in range(len(batch))]);collected[key].append(batch_curves[key])
                for j,start in enumerate(batch):cases.append(dict(condition=label,seed=seed+1,start=int(start),food=float(result[j,2]),visited=float(result[j,3]),reachable_visited=float(batch_curves['reachable_visited'][j,-1]),coverage_percent=float(100*batch_curves['reachable_visited'][j,-1]/accessible),completion_seconds=float(result[j,1])))
                with (args.output/'encounters.csv').open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(cases[0]));writer.writeheader();writer.writerows(cases)
                print(f'{label} | controller {seed+1}/{n} | starts {min(offset+len(batch),args.starts)}/{args.starts} | elapsed {time.perf_counter()-started:.1f}s',flush=True)
            arrays={key:np.concatenate(parts,axis=0) for key,parts in collected.items()}
            raw[f'{seed}_{int(enabled)}_food']=arrays['food'];raw[f'{seed}_{int(enabled)}_visited']=arrays['visited'];raw[f'{seed}_{int(enabled)}_reachable']=arrays['reachable_visited']
            for i,seconds in enumerate(times):curves.append(dict(condition=label,seed=seed+1,seconds=float(seconds),food=float(arrays['food'][:,i].mean()),visited=float(arrays['visited'][:,i].mean()),coverage_percent=float(100*arrays['reachable_visited'][:,i].mean()/accessible)))
    with (args.output/'seed-curves.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(curves[0]));writer.writeheader();writer.writerows(curves)
    np.savez_compressed(args.output/'encounter-curves.npz',times=times,**raw)
    summary={};values_by={};fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for label,color in [('Recovery off','#ba6954'),('Recovery on','#188b87')]:
        for ax,key,title in zip(axes,('food','visited'),('Pellets collected','Grid tiles visited')):
            values=np.array([[r[key] for r in curves if r['condition']==label and r['seed']==seed+1] for seed in range(n)])
            values_by[label,key]=values;mean,low,high=seed_mean_interval(values)
            ax.plot(times,mean,color=color,lw=2.4,label=label);ax.fill_between(times,low,high,color=color,alpha=.18)
            ax.set(title=title,xlabel='Simulated seconds',ylabel='Pellets' if key=='food' else 'Tiles',ylim=(0,count if key=='food' else grid**2));ax.legend()
            summary.setdefault(label,{})[key]=dict(initial=float(mean[0]),final_mean=float(mean[-1]),final_ci95=[float(low[-1]),float(high[-1])],controller_means=values[:,-1].tolist())
        coverage=np.array([[r['coverage_percent'] for r in curves if r['condition']==label and r['seed']==seed+1] for seed in range(n)])
        m,l,h=seed_mean_interval(coverage)
        summary[label]['coverage_percent']=dict(final_mean=float(m[-1]),final_ci95=[float(l[-1]),float(h[-1])])
    fig.suptitle(f'Stuck recovery: {n*args.starts} paired encounters per condition, {n} saved controller seeds\nMeans and pointwise 95% bootstrap CIs across controller seeds; no retraining')
    fig.savefig(args.output/'recovery-means-ci.png',dpi=170);plt.close(fig)
    paired={};fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,key,title in zip(axes,('food','visited'),('Additional pellets with recovery','Additional tiles with recovery')):
        differences=values_by['Recovery on',key]-values_by['Recovery off',key];mean,low,high=seed_mean_interval(differences)
        ax.axhline(0,color='gray',ls='--',lw=1);ax.plot(times,mean,color='#5368a0',lw=2.4);ax.fill_between(times,low,high,color='#5368a0',alpha=.2)
        ax.set(title=title,xlabel='Simulated seconds',ylabel='Recovery on minus off')
        paired[key]=dict(final_difference=float(mean[-1]),ci95=[float(low[-1]),float(high[-1])],controller_differences=differences[:,-1].tolist())
    fig.suptitle('Paired recovery effect: same controllers and starts\nPointwise 95% seed-bootstrap CIs; positive favours recovery')
    fig.savefig(args.output/'paired-differences.png',dpi=170);plt.close(fig)
    summary['paired_differences']=paired
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    (args.output/'applied-instructions.json').write_text(json.dumps(HYBRID_INSTRUCTIONS,indent=2))
    (args.output/'design.json').write_text(json.dumps(dict(source=str(args.run),controller_seeds=n,starts_per_controller=args.starts,encounters_per_condition=n*args.starts,seconds=args.seconds,accessible_grid_tiles=accessible,total_grid_tiles=grid**2,food=count,checkpoint_seconds=1,starts='Fresh deterministic IDs 92000000 + controller_index*10000 + [0..starts-1], not used for training or the earlier three replays',confidence='10000 percentile bootstrap resamples of entire controller-mean trajectories; paired conditions resampled together; pointwise 95% intervals, not simultaneous',notes=['Same saved weights and starts; only programmed recovery is toggled. No new evolutionary training.','Multiple encounters per controller are clustered, not counted as independent evolutionary seeds.','Only five independent controllers exist in the source experiment; confidence estimates remain based on that small controller sample.','Coverage percentage counts reachable grid tiles entered by body centre; visited counts include starting tile.','Inference is conditional on this single terrain map and the saved 20-generation controllers.'],champions_sha256=hashlib.sha256(champion_path.read_bytes()).hexdigest(),elapsed_seconds=time.perf_counter()-started),indent=2))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
