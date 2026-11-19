"""Run matched fixed-opponent/co-evolution treatments and shared benchmarks."""
from src.paths import ROOT
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generations',type=int,default=50)
    p.add_argument('--seeds',type=int,default=3)
    p.add_argument('--eval-every',type=int,default=5)
    p.add_argument('--seed-offset',type=int,default=0)
    p.add_argument('--benchmark-reference',type=Path)
    p.add_argument('--output',type=Path,default=Path('results/opponent-comparison-50'))
    args=p.parse_args()
    if args.seeds<2 or min(args.generations,args.eval_every)<1:p.error('Use at least two seeds and positive generation/evaluation counts')
    if args.benchmark_reference:
        current=set(range(args.seed_offset,args.seed_offset+args.seeds))
        for mode in ('coevolution','fixed'):
            path=args.benchmark_reference/mode
            reference=json.loads((path/'design.json').read_text())
            if not (path/'history.json').exists():p.error('Reference run needs history.json')
            previous=set(range(reference.get('seed_offset',0),reference.get('seed_offset',0)+reference['seeds']))
            if current & previous:p.error('Use --seed-offset to separate new seeds from benchmark seeds')
    root=ROOT
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    common=['--arena','empty','--food-encoding','unit','--conditions','equal','--training-starts','8','--generations',str(args.generations),'--seeds',str(args.seeds),'--eval-every',str(args.eval_every),'--seed-offset',str(args.seed_offset)]
    for i,mode in enumerate(('coevolution','fixed'),1):
        print(f'[{i}/3] Training {mode}',flush=True)
        subprocess.run([sys.executable,'-u','-B','-m','src.simulation.vision_pilot',*common,'--opponent-mode',mode,'--output',str(output/mode)],cwd=root,check=True)
    print('[3/3] Evaluating both treatments against the same fixed panel',flush=True)
    evaluation=[sys.executable,'-u','-B','-m','src.analysis.evaluate_pursuit_progress',str(output/'fixed'),'--compare-opponents',str(output/'coevolution')]
    if args.benchmark_reference:evaluation+=['--benchmark-reference',str(args.benchmark_reference.resolve())]
    subprocess.run(evaluation,cwd=root,check=True)
    print(f'Finished. Figures and data: {output / "fixed" / "opponent-comparison"}',flush=True)


if __name__=='__main__':main()
