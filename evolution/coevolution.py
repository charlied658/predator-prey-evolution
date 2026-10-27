"""Competitive predator/prey evolution, archives, historical tournaments, and replays."""
import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from experiment import breed, write_json
from godot_interface import evaluate_batch
from config import GODOT_PATH, PROJECT_PATH

OBSTACLES = [[0.0, 0.0, 2.0], [-6.0, 5.0, 2.0], [6.0, -5.0, 2.0]]
GENOME_LENGTH = 218


def new_genome(rng):
    weights = [rng.uniform(-1, 1) for _ in range(GENOME_LENGTH)]
    for neuron in range(12):
        base = neuron * 16
        weights[base + 4] = 0.0
        weights[base + 15] = 0.0
        # Small initial wall/obstacle influence; all sensor weights remain evolvable.
        for index in range(5, 15):
            weights[base + index] *= 0.15
    weights[204] = weights[217] = 0.0
    return weights


def free_position(rng, obstacles):
    while True:
        point = [rng.uniform(-12, 12), rng.uniform(-12, 12)]
        if all(math.dist(point, obstacle[:2]) > obstacle[2] + 0.8 for obstacle in obstacles):
            return point


def scenarios(seed, count, obstacles):
    rng = random.Random(seed)
    result = []
    while len(result) < count:
        p, q = free_position(rng, obstacles), free_position(rng, obstacles)
        if 5 <= math.dist(p, q) <= 20:
            result.append({'predator_start': p, 'prey_start': q, 'obstacles': obstacles,
                           'predator_sensor_mode': 'ecology', 'prey_sensor_mode': 'ecology'})
    return result


def pair(predator, prey, case, label='', trace=False):
    return {**case, 'predator_genome': predator, 'prey_genome': prey,
            'label': label, 'record_trace': trace}


def role_score(row, role):
    if role == 'predator':
        if row['captured']:
            return row['predator_fitness']
        closeness = max(-1, 1 - row['mean_distance'] / max(row['initial_distance'], 1))
        return 10 * closeness + 5 * row['approach_alignment']
    # Survival is primary; bounded separation breaks equal-survival ties.
    return row['time'] + 0.5 * min(row['mean_distance'] / 30, 1)


def select_opponents(population, archive, count, rng):
    current_count = (count + 1) // 2
    current = [population[rng.randrange(len(population))] for _ in range(current_count)]
    historical = [archive[rng.randrange(len(archive))]['genome'] for _ in range(count - current_count)]
    return current + historical


def evaluate_role(population, opponents, cases, role, workers):
    encounters = []
    for individual in population:
        for opponent, case in zip(opponents, cases):
            encounters.append(pair(individual, opponent, case) if role == 'predator'
                              else pair(opponent, individual, case))
    rows = evaluate_batch(encounters, workers)
    width = len(cases)
    groups = [rows[i:i + width] for i in range(0, len(rows), width)]
    scores = [statistics.mean(role_score(row, role) for row in group) for group in groups]
    winner = max(range(len(scores)), key=scores.__getitem__)
    return scores, population[winner][:], groups[winner]


def row_metrics(rows):
    result = {'capture_rate': statistics.mean(r['captured'] for r in rows),
              'mean_survival': statistics.mean(r['time'] for r in rows)}
    for role in ['predator', 'prey']:
        for name in ['path_length', 'turn_angle', 'blocked_time', 'obstacle_time']:
            key = f'{role}_{name}'
            result[key] = statistics.mean(r[key] for r in rows)
    return result


def restore_tuple(value):
    return tuple(restore_tuple(x) for x in value) if isinstance(value, list) else value


def train(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in ['resume', 'command']}
    sources = [Path(__file__), Path(__file__).with_name('experiment.py'),
               Path(__file__).with_name('godot_interface.py'),
               Path(PROJECT_PATH) / 'scripts/agent.gd', Path(PROJECT_PATH) / 'scripts/simulation.gd',
               Path(PROJECT_PATH) / 'world.tscn', Path(PROJECT_PATH) / 'project.godot']
    provenance = {'python': sys.version, 'source_sha256': {
        str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    manifest = output / 'config.json'
    if manifest.exists():
        if not args.resume:
            raise ValueError('Output already exists. Choose a new directory or use --resume.')
        if json.loads(manifest.read_text()) != config or json.loads((output / 'provenance.json').read_text()) != provenance:
            raise ValueError('Resume requires identical settings, Python version, and simulation sources.')
    elif args.resume:
        raise ValueError('No existing run to resume.')
    else:
        write_json(manifest, config)
        write_json(output / 'provenance.json', provenance)
    obstacles = OBSTACLES if args.arena == 'obstacles' else []
    rng = random.Random(args.seed)
    predators = [new_genome(rng) for _ in range(args.population)]
    prey = [new_genome(rng) for _ in range(args.population)]
    pred_archive = [{'generation': 0, 'genome': predators[0]}]
    prey_archive = [{'generation': 0, 'genome': prey[0]}]
    history = []
    start = 1
    checkpoint = output / 'checkpoint.json'
    if args.resume and checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        predators, prey = state['predators'], state['prey']
        pred_archive, prey_archive = state['predator_archive'], state['prey_archive']
        history = state['history']
        rng.setstate(restore_tuple(state['rng_state']))
        start = state['next_generation']
    reference = scenarios(args.seed + 7_000_000, args.reference_cases, obstacles)
    write_json(output / 'reference_scenarios.json', reference)
    (output / 'generations').mkdir(exist_ok=True)
    write_json(output / 'generations/generation-0000.json', {
        'generation': 0, 'predator_genome': pred_archive[0]['genome'], 'prey_genome': prey_archive[0]['genome']})
    for generation in range(start, args.generations + 1):
        cases = scenarios(args.seed * 100_003 + generation, args.episodes, obstacles)
        pred_opponents = select_opponents(prey, prey_archive, args.episodes, rng)
        prey_opponents = select_opponents(predators, pred_archive, args.episodes, rng)
        # Both populations face pre-reproduction opponents; append archives only after both evaluations.
        pred_scores, pred_champion, pred_rows = evaluate_role(predators, pred_opponents, cases, 'predator', args.workers)
        prey_scores, prey_champion, prey_rows = evaluate_role(prey, prey_opponents, cases, 'prey', args.workers)
        anchor_rows = evaluate_batch(
            [pair(pred_champion, prey_archive[0]['genome'], case) for case in reference]
            + [pair(pred_archive[0]['genome'], prey_champion, case) for case in reference]
            + [pair(pred_champion, prey_champion, case) for case in reference], args.workers)
        n = len(reference)
        head_to_head = anchor_rows[2 * n:]
        history.append({'generation': generation,
                        'training_encounters': generation * 2 * args.population * args.episodes,
                        'reference_encounters': generation * 3 * n,
                        'predator_mean_fitness': statistics.mean(pred_scores),
                        'prey_mean_fitness': statistics.mean(prey_scores),
                        'predator_champion_training_capture_rate': row_metrics(pred_rows)['capture_rate'],
                        'prey_champion_training_survival': row_metrics(prey_rows)['mean_survival'],
                        'predator_vs_initial_prey_capture_rate': row_metrics(anchor_rows[:n])['capture_rate'],
                        'initial_predator_vs_prey_capture_rate': row_metrics(anchor_rows[n:2*n])['capture_rate'],
                        **{f'head_to_head_{k}': v for k, v in row_metrics(head_to_head).items()}})
        snapshot = {'generation': generation, 'predator_genome': pred_champion,
                    'prey_genome': prey_champion, 'metrics': history[-1]}
        write_json(output / f'generations/generation-{generation:04d}.json', snapshot)
        pred_archive.append({'generation': generation, 'genome': pred_champion})
        prey_archive.append({'generation': generation, 'genome': prey_champion})
        predators = breed(predators, pred_scores, rng, crossover=False)
        prey = breed(prey, prey_scores, rng, crossover=False)
        write_json(checkpoint, {'next_generation': generation + 1, 'predators': predators, 'prey': prey,
                               'predator_archive': pred_archive, 'prey_archive': prey_archive,
                               'rng_state': rng.getstate(), 'history': history})
        with (output / 'history.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        print(f"Generation {generation}/{args.generations}: current capture={row_metrics(head_to_head)['capture_rate']:.0%}, "
              f"predator vs original prey={history[-1]['predator_vs_initial_prey_capture_rate']:.0%}", flush=True)
    # Rebuild this derived file even when a crash occurred just after checkpointing.
    with (output / 'history.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    analyze(args.output, args.matrix_every, args.matrix_cases, args.workers)


def tournament(output, every, case_count, workers):
    config = json.loads((output / 'config.json').read_text())
    snapshots = [json.loads(p.read_text()) for p in sorted((output / 'generations').glob('generation-*.json'))]
    last = snapshots[-1]['generation']
    selected = [s for s in snapshots if s['generation'] % every == 0 or s['generation'] == last]
    cases = scenarios(config['seed'] + 8_000_000, case_count, OBSTACLES if config['arena'] == 'obstacles' else [])
    encounters = [pair(p['predator_genome'], q['prey_genome'], case)
                  for p in selected for q in selected for case in cases]
    rows = evaluate_batch(encounters, workers)
    records = []
    offset = 0
    for predator in selected:
        for prey in selected:
            group = rows[offset:offset + case_count]
            offset += case_count
            records.append({'predator_generation': predator['generation'], 'prey_generation': prey['generation'],
                            **row_metrics(group)})
    return records, cases, snapshots


def analyze(output, every, case_count, workers):
    records, cases, snapshots = tournament(output, every, case_count, workers)
    write_json(output / 'historical_tournament.json', {'scenarios': cases, 'results': records})
    with (output / 'historical_tournament.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    final = snapshots[-1]
    first = snapshots[0]
    playlist = []
    for p, q in [(first, first), (final, first), (first, final), (final, final)]:
        for i, case in enumerate(cases[:3]):
            playlist.append(pair(p['predator_genome'], q['prey_genome'], case,
                                 f"Predator generation {p['generation']} vs prey generation {q['generation']} | scenario {i}"))
    write_json(output / 'replay.json', {'encounters': playlist})
    # Short trajectories for every saved generation use the same scenario (descriptive, not selected highlights).
    trace_cases = [pair(s['predator_genome'], s['prey_genome'], cases[0],
                       f"Generation {s['generation']}", True) for s in snapshots]
    traced = evaluate_batch(trace_cases, workers)
    write_json(output / 'trajectories.json', [{'generation': s['generation'], 'case': case, 'result': row}
                                            for s, case, row in zip(snapshots, trace_cases, traced)])
    write_dashboard(output, records, traced, snapshots, cases[0]['obstacles'])
    print(f"Saved historical tournament, trajectories, dashboard, and replay to {output}", flush=True)


def write_dashboard(output, records, traced, snapshots, obstacles):
    # Fully local interactive report; escape '<' before embedding data in JavaScript.
    payload = json.dumps({'matrix': records, 'traces': traced,
                          'generations': [s['generation'] for s in snapshots], 'obstacles': obstacles}).replace('<', '\\u003c')
    template = '''<!doctype html><html><head><meta charset="utf-8"><title>Coevolution history</title>
<style>body{font:16px system-ui;background:#101827;color:#e2e8f0;margin:30px auto;max-width:1080px}section{background:#192538;padding:22px;margin:20px 0;border-radius:12px}canvas{background:#edf2f7;max-width:100%}label{display:block;margin:14px 0}select,input{margin-left:12px}p{line-height:1.5}#matrix{display:grid;gap:2px;width:max-content;max-width:100%;overflow:auto}.cell{padding:10px;min-width:34px;text-align:center;color:#111827}.axis{padding:10px;font-size:13px}button{padding:8px;margin-right:12px}small{color:#b5c4d8}</style></head><body>
<h1>Predator–prey coevolution</h1><p>Red pursues; green evades. Each generation is a snapshot of two evolving populations.</p>
<section><h2>Behavior through time</h2><label>Generation <select id="generation"></select></label><label>Time <input id="time" type="range" min="0" max="100" value="100"><span id="clock"></span></label><canvas id="arena" width="620" height="620"></canvas><p id="details"></p><small>The same starting scenario is used for every generation. Traces show one encounter, not population-wide behavior. Circle outlines include agent clearance.</small></section>
<section><h2>Historical opponents</h2><p>Rows: predator generation. Columns: prey generation. Cells: capture rate across identical tournament scenarios. Read across a row to inspect one predator against past and future prey.</p><div id="matrix"></div><p>A strong diagonal alone does not establish progress. Beating old opponents while losing to newer ones is consistent with an arms race; repeated reversals can indicate counterstrategy cycles.</p><small>Small scenario counts are noisy. This tournament describes this run and is not an independent statistical test.</small></section>
<script>const data=PAYLOAD;const gen=document.querySelector('#generation'),time=document.querySelector('#time'),canvas=document.querySelector('#arena'),ctx=canvas.getContext('2d');data.generations.forEach((g,i)=>gen.add(new Option(g,i)));gen.value=data.generations.length-1;const point=p=>[(p[0]+15)*620/30,(15-p[1])*620/30];function draw(){const r=data.traces[+gen.value],t=r.trajectory||[],end=Math.max(1,Math.ceil(t.length*+time.value/100));ctx.clearRect(0,0,620,620);ctx.strokeStyle='#718096';ctx.strokeRect(620/30,620/30,620*28/30,620*28/30);data.obstacles.forEach(o=>{const p=point(o);ctx.beginPath();ctx.arc(...p,o[2]*620/30,0,Math.PI*2);ctx.fillStyle='#64748b';ctx.fill();ctx.beginPath();ctx.arc(...p,(o[2]+.5)*620/30,0,Math.PI*2);ctx.stroke()});['predator','prey'].forEach((role,index)=>{ctx.strokeStyle=index?'#15803d':'#dc2626';ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=2;ctx.beginPath();t.slice(0,end).forEach((s,i)=>{const p=point(s[role]);i?ctx.lineTo(...p):ctx.moveTo(...p)});ctx.stroke();if(t.length){const p=point(t[Math.min(end-1,t.length-1)][role]);ctx.beginPath();ctx.arc(...p,7,0,Math.PI*2);ctx.fill()}});document.querySelector('#clock').textContent=t.length?' '+t[Math.min(end-1,t.length-1)].time.toFixed(1)+' s':'';document.querySelector('#details').textContent=(r.captured?'Captured':'Prey survived')+' at '+r.time.toFixed(1)+'s. Turns: predator '+(r.predator_turn_angle/(2*Math.PI)).toFixed(1)+' total rotations; prey '+(r.prey_turn_angle/(2*Math.PI)).toFixed(1)+'.';}gen.onchange=draw;time.oninput=draw;draw();const generations=[...new Set(data.matrix.map(r=>r.predator_generation))],matrix=document.querySelector('#matrix');matrix.style.gridTemplateColumns=`repeat(${generations.length+1},auto)`;function cell(text,css,bg){const d=document.createElement('div');d.className=css;d.textContent=text;if(bg)d.style.background=bg;matrix.append(d)}cell('P / Q','axis');generations.forEach(g=>cell(g,'axis'));generations.forEach(p=>{cell(p,'axis');generations.forEach(q=>{const r=data.matrix.find(r=>r.predator_generation===p&&r.prey_generation===q);cell(Math.round(r.capture_rate*100)+'%','cell',`hsl(${120*(1-r.capture_rate)} 55% 70%)`)})});</script></body></html>'''
    (output / 'dashboard.html').write_text(template.replace('PAYLOAD', payload), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    training = sub.add_parser('train')
    training.add_argument('--output', type=Path, default=Path('results/coevolution'))
    training.add_argument('--seed', type=int, default=200)
    training.add_argument('--population', type=int, default=40)
    training.add_argument('--generations', type=int, default=60)
    training.add_argument('--episodes', type=int, default=8)
    training.add_argument('--reference-cases', type=int, default=8)
    training.add_argument('--arena', choices=['open', 'obstacles'], default='obstacles')
    training.add_argument('--workers', type=int, default=2)
    training.add_argument('--matrix-every', type=int, default=10)
    training.add_argument('--matrix-cases', type=int, default=12)
    training.add_argument('--resume', action='store_true')
    analysis = sub.add_parser('analyze')
    analysis.add_argument('output', type=Path)
    analysis.add_argument('--matrix-every', type=int, default=10)
    analysis.add_argument('--matrix-cases', type=int, default=12)
    analysis.add_argument('--workers', type=int, default=2)
    replay_parser = sub.add_parser('replay')
    replay_parser.add_argument('output', type=Path)
    replay_parser.add_argument('--predator-generation', type=int)
    replay_parser.add_argument('--prey-generation', type=int)
    replay_parser.add_argument('--scenarios', type=int, default=3)
    args = parser.parse_args()
    for key in ['population', 'generations', 'episodes', 'reference_cases', 'workers', 'matrix_every', 'matrix_cases', 'scenarios']:
        if hasattr(args, key) and getattr(args, key) < 1:
            parser.error(f'{key} must be positive')
    if args.command == 'train' and (args.population < 3 or args.episodes < 2):
        parser.error('Use at least 3 individuals and 2 episodes for current/archive opponent mixing.')
    try:
        if args.command == 'train':
            train(args)
        elif args.command == 'analyze':
            analyze(args.output, args.matrix_every, args.matrix_cases, args.workers)
        else:
            replay(args)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


def replay(args):
    output = args.output
    if args.predator_generation is None and args.prey_generation is None:
        path = output / 'replay.json'
        subprocess.run([GODOT_PATH, '--path', PROJECT_PATH, '--', '--replay', str(path.resolve())], check=True)
        return
    config = json.loads((output / 'config.json').read_text())
    snapshots = sorted((output / 'generations').glob('generation-*.json'))
    latest = json.loads(snapshots[-1].read_text())['generation']
    pg = latest if args.predator_generation is None else args.predator_generation
    qg = latest if args.prey_generation is None else args.prey_generation
    p = json.loads((output / f'generations/generation-{pg:04d}.json').read_text())
    q = json.loads((output / f'generations/generation-{qg:04d}.json').read_text())
    cases = scenarios(config['seed'] + 8_000_000, args.scenarios, OBSTACLES if config['arena'] == 'obstacles' else [])
    encounters = [pair(p['predator_genome'], q['prey_genome'], case,
                       f'Predator generation {pg} vs prey generation {qg} | scenario {i}') for i, case in enumerate(cases)]
    with tempfile.TemporaryDirectory(prefix='coevolution-replay-') as directory:
        path = Path(directory) / 'replay.json'
        write_json(path, {'encounters': encounters})
        subprocess.run([GODOT_PATH, '--path', PROJECT_PATH, '--', '--replay', str(path.resolve())], check=True)


if __name__ == '__main__':
    main()
