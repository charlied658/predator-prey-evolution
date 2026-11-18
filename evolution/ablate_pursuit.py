"""Matched one-factor pilot of simplified predator-prey mechanics."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CONDITIONS = ('baseline', 'speed', 'turning', 'acceleration', 'obstacles',
              'food', 'hunger', 'joints', 'swimming')
INPUTS, HIDDEN, OUTPUTS = 18, 6, 4
GENES = (INPUTS + 1) * HIDDEN + (HIDDEN + 1) * OUTPUTS
DT, STEPS, RADIUS = .05, 240, .18
OBSTACLES = np.array([[-1.5, 1., .6], [1.5, -1., .6], [0, 2, .55], [0, -2, .55]])
PELLETS = np.array([[-2., -2.], [2., 2.], [-2., 2.], [2., -2.]])


def weights(genomes):
    cut = (INPUTS + 1) * HIDDEN
    return genomes[:, :cut].reshape(-1, INPUTS + 1, HIDDEN), genomes[:, cut:].reshape(-1, HIDDEN + 1, OUTPUTS)


def rotate(v, heading):
    c, s = np.cos(heading), np.sin(heading)
    return np.stack((v[..., 0] * c + v[..., 1] * s, -v[..., 0] * s + v[..., 1] * c), -1)


def episodes(a, b, seed_ids, condition, environment=None, trajectory=None):
    obstacles = OBSTACLES if environment is None else environment['obstacles']
    food_sites = PELLETS if environment is None else environment['pellets']
    has_obstacles = condition == 'obstacles' or environment is not None
    has_food = condition == 'food' or environment is not None
    count = len(a)
    # Per-start construction keeps random positions matched across batches.
    starts = np.array([np.random.default_rng(int(seed)).uniform(-1, 1, 5) for seed in seed_ids])
    angle = starts[:, 0] * np.pi
    distance = 1.2 + .3 * (starts[:, 1] + 1)
    offset = np.stack((np.cos(angle), np.sin(angle)), -1) * distance[:, None] / 2
    pos = np.stack((-offset, offset), 1)
    if environment is not None:
        pos = np.stack((np.c_[-2 + .3*starts[:,0], -.8 + .2*starts[:,1]],
                        np.c_[-2 + .3*starts[:,0], .8 + .2*starts[:,1]]), axis=1)
        if environment['rotate']:
            pos = pos[..., ::-1].copy()
    heading = starts[:, 2:4] * np.pi
    velocity = np.zeros_like(pos)
    bend = np.zeros((count, 2))
    energy = np.ones((count, 2))
    active = np.ones(count, bool)
    caught = np.zeros(count, bool)
    end = np.full(count, STEPS * DT)
    food = np.zeros(count)
    pellets = np.ones((count, len(food_sites)), bool)
    crossings = np.zeros((count, 2))
    if trajectory is not None:
        trajectory.append(pos.copy())
    speed_sum = np.zeros(count)
    turn_sum = np.zeros(count)
    nearest = np.linalg.norm(pos[:, 0] - pos[:, 1], axis=1)
    nets = [weights(a), weights(b)]
    articulated = condition in ('joints', 'swimming')
    samples = np.linspace(0, .65, 5)

    def valid(head, h, joint):
        points = head[:, :, None, :]
        if articulated:
            axis = np.stack((np.cos(h), np.sin(h)), -1)
            tail_axis = np.stack((np.cos(h + joint), np.sin(h + joint)), -1)
            front = head[:, :, None, :] - samples[None, None, :, None] * axis[:, :, None, :]
            tail = head[:, :, None, :] - .65 * axis[:, :, None, :] - samples[None, None, :, None] * tail_axis[:, :, None, :]
            points = np.concatenate((front, tail), axis=2)
        clear = np.all(np.abs(points) < 3.8, axis=(2, 3))
        if has_obstacles:
            distance = np.linalg.norm(points[:, :, :, None, :] - obstacles[None, None, None, :, :2], axis=-1)
            clear &= np.all(distance > obstacles[None, None, None, :, 2] + RADIUS, axis=(2, 3))
        return clear

    for tick in range(STEPS):
        if not active.any():
            break
        outputs = []
        for side in range(2):
            other = 1 - side
            relative = rotate(pos[:, other] - pos[:, side], heading[:, side]) / 8
            relative_velocity = rotate(velocity[:, other] - velocity[:, side], heading[:, side]) / 4
            own_velocity = rotate(velocity[:, side], heading[:, side]) / 2
            walls = np.concatenate(((4 - pos[:, side]) / 8, (4 + pos[:, side]) / 8), axis=1)
            if has_obstacles:
                offset = obstacles[None, :, :2] - pos[:, side, None, :]
                radii = obstacles[None, :, 2] + RADIUS
                for ray, (dx, dy) in enumerate(((1, 0), (0, 1), (-1, 0), (0, -1))):
                    along = offset[:, :, 0] * dx + offset[:, :, 1] * dy
                    perpendicular = offset[:, :, 0] * dy - offset[:, :, 1] * dx
                    hit = along - np.sqrt(np.maximum(0, radii**2 - perpendicular**2))
                    hit[(np.abs(perpendicular) > radii) | (hit < 0)] = np.inf
                    walls[:, ray] = np.minimum(walls[:, ray], hit.min(axis=1) / 8)
            food_direction = np.zeros((count, 2))
            if has_food:
                distances = np.linalg.norm(pos[:, side, None] - food_sites[None], axis=-1)
                distances[~pellets] = np.inf
                index = distances.argmin(axis=1)
                delta = food_sites[index] - pos[:, side]
                delta[~pellets.any(axis=1)] = 0
                food_direction = rotate(delta, heading[:, side]) / 8
            clock = np.tile([np.sin(tick * DT * 7.5), np.cos(tick * DT * 7.5)], (count, 1))
            obs = np.concatenate((relative, relative_velocity, own_velocity, walls,
                                  food_direction, energy[:, side, None], bend[:, side, None], clock,
                                  np.full((count, 1), tick / STEPS), np.ones((count, 1))), axis=1)
            w, v = nets[side]
            hidden = np.tanh(np.einsum('ni,nij->nj', np.c_[obs, np.ones(count)], w))
            outputs.append(np.tanh(np.einsum('ni,nij->nj', np.c_[hidden, np.ones(count)], v)))
        command = np.stack(outputs, 1)
        desired_turn = np.arctan2(command[:, :, 1], command[:, :, 0])
        new_heading = heading + desired_turn
        if condition in ('turning', 'joints', 'swimming'):
            limit = np.array([3., 3.])
            new_heading = heading + np.clip(desired_turn, -limit * DT, limit * DT)
        if condition == 'swimming':
            new_heading = heading + np.clip(desired_turn, -.7 * DT, .7 * DT) * np.clip(np.linalg.norm(velocity, axis=-1), 0, 1)
        new_bend = bend + np.clip(command[:, :, 3] - bend, -2.5 * DT, 2.5 * DT)
        new_bend *= articulated
        pose_ok = valid(pos, new_heading, new_bend)
        new_heading = np.where(pose_ok, new_heading, heading)
        new_bend = np.where(pose_ok, new_bend, bend)
        forward = np.stack((np.cos(new_heading), np.sin(new_heading)), -1)
        max_speed = np.array([2.3, 2.]) if condition == 'speed' else np.array([2., 2.])
        target = forward * max_speed[None, :, None] * (command[:, :, 2:3] + 1) / 2
        if condition == 'hunger':
            energy = np.maximum(0, energy - DT * .06)
            target *= (.3 + .7 * energy[:, :, None])
        if condition == 'acceleration':
            delta = target - velocity
            velocity += delta * np.minimum(1, 3 * DT / np.maximum(np.linalg.norm(delta, axis=-1, keepdims=True), 1e-9))
        elif condition == 'swimming':
            stroke = np.abs(new_bend - bend) / DT
            velocity = velocity * np.exp(-1.8 * DT) + forward * np.minimum(8, 1.4 * stroke**2)[:, :, None] * DT
            velocity *= np.minimum(1, 2 / np.maximum(np.linalg.norm(velocity, axis=-1, keepdims=True), 1e-9))
        else:
            velocity = target
        candidate = pos + velocity * DT
        ok = valid(candidate, new_heading, new_bend)
        candidate = np.where(ok[:, :, None], candidate, pos)
        previous = pos.copy()
        pos = np.where(active[:, None, None], candidate, pos)
        if environment is not None:
            axis = int(environment['rotate'])
            crossings += ((previous[:,:,axis] * pos[:,:,axis]) < 0) * active[:,None]
        if trajectory is not None:
            trajectory.append(pos.copy())
        turn_sum += np.abs(np.remainder(new_heading[:, 1] - heading[:, 1] + np.pi, 2*np.pi)-np.pi) * active
        heading = np.where(active[:, None], new_heading, heading)
        bend = np.where(active[:, None], new_bend, bend)
        speed_sum += np.linalg.norm(pos[:, 1] - previous[:, 1], axis=1) * active
        rel0 = previous[:, 0] - previous[:, 1]
        rel1 = pos[:, 0] - pos[:, 1]
        travel = rel1 - rel0
        fraction = np.clip(-np.sum(rel0 * travel, axis=1) / np.maximum(np.sum(travel**2, axis=1), 1e-12), 0, 1)
        distance = np.linalg.norm(rel0 + fraction[:, None] * travel, axis=1)
        nearest = np.minimum(nearest, distance)
        tag = active & (distance <= 2 * RADIUS)
        caught |= tag
        end[tag] = (tick + 1) * DT
        active[tag] = False
        if has_food:
            close = np.linalg.norm(pos[:, 1, None] - food_sites[None], axis=-1) < .35
            eaten = close & pellets & active[:, None]
            food += eaten.sum(axis=1)
            pellets[eaten] = False
    fraction = end / (STEPS * DT)
    predator = caught * (2 - fraction) + (~caught) * .15 * np.exp(-nearest)
    prey = fraction - caught
    if has_food:
        prey += .25 * food
    result = np.c_[predator, prey, caught, fraction, speed_sum / np.maximum(end, DT),
                   turn_sum / np.maximum(end, DT), food]
    return np.c_[result, crossings] if environment is not None else result


def mutate(parent, rng):
    child = parent.copy()
    indices = rng.choice(GENES, 8, replace=False)
    child[indices] += rng.normal(0, .35, 8)
    return child


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/pursuit-elements'))
    parser.add_argument('--generations', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.generations < 9 or args.repeats < 1:
        parser.error('Use at least nine generations and one repeat')
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    timings = []
    final = {}
    for repeat in range(args.repeats):
        rng = np.random.default_rng(510 + repeat)
        initial = rng.normal(0, .45, (4, GENES))
        training = np.arange(4) + 1000 * repeat
        testing = np.arange(24) + 100000 + repeat * 100
        # All conditions get identical initial genomes, starts, and mutation RNG.
        for condition in CONDITIONS:
            streams = [np.random.default_rng(9000 + repeat*2 + side) for side in range(2)]
            champions = [initial[0].copy(), initial[0].copy()]
            populations = [initial.copy(), initial.copy()]
            started = time.perf_counter()
            checkpoints = {}
            for generation in range(args.generations + 1):
                if generation:
                    pairs = [(a, b) for a in populations[0] for b in populations[1]]
                    a = np.repeat(np.array([x for x, _ in pairs]), len(training), axis=0)
                    b = np.repeat(np.array([y for _, y in pairs]), len(training), axis=0)
                    result = episodes(a, b, np.tile(training, len(pairs)), condition).reshape(4,4,len(training),7)
                    scores = [result[:,:,:,0].mean(axis=(1,2)), result[:,:,:,1].mean(axis=(0,2))]
                    champions = [p[s.argmax()].copy() for p,s in zip(populations,scores)]
                    populations = [np.array([g]+[mutate(g,r) for _ in range(3)]) for g,r in zip(champions,streams)]
                if generation in (0,4,8,args.generations):
                    # Each role against the same pre-training policies.
                    role_results = []
                    for side in range(2):
                        g = np.repeat(champions[side][None], len(testing), axis=0)
                        frozen = np.repeat(initial[0][None], len(testing), axis=0)
                        role_results.append(episodes(g if side==0 else frozen, frozen if side==0 else g, testing, condition))
                    row = {'repeat':repeat,'condition':condition,'generation':generation,
                           'predator_capture':role_results[0][:,2].mean(),
                           'prey_escape':1-role_results[1][:,2].mean(),
                           'prey_survival':role_results[1][:,3].mean(),
                           'prey_speed':role_results[1][:,4].mean(),
                           'prey_turn_rate':role_results[1][:,5].mean(),
                           'prey_food':role_results[1][:,6].mean()}
                    rows.append(row)
                    checkpoints[generation] = [g.tolist() for g in champions]
            timings.append({'repeat':repeat,'condition':condition,'seconds':time.perf_counter()-started})
            final[f'{repeat}/{condition}'] = checkpoints
            print(f'Run {repeat+1}/{args.repeats} {condition:13} {timings[-1]["seconds"]:.1f}s | capture {row["predator_capture"]:.0%} escape {row["prey_escape"]:.0%}',flush=True)
        with (args.output/'measurements.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        (args.output/'timing.json').write_text(json.dumps(timings,indent=2),encoding='utf-8')
    (args.output/'champions.json').write_text(json.dumps(final),encoding='utf-8')
    plot(args.output,rows,timings,args.generations)
    (args.output/'design.json').write_text(json.dumps({'conditions':CONDITIONS,'repeats':args.repeats,'generations':args.generations,
        'genes':GENES,'dt':DT,'duration':STEPS*DT,'population':4,'training_starts':4,'test_starts':24,
        'mechanics': {
            'baseline': 'Equal max speed 2; circle heads; instantaneous steering and acceleration; bounded arena.',
            'speed': 'Predator max speed 2.3; prey 2.0.',
            'turning': 'Both turn at at most 3 radians/second.',
            'acceleration': 'Both accelerate at at most 3 distance units/second squared.',
            'obstacles': 'Four fixed circles; collisions and four directional clearance readings.',
            'food': 'Four fixed pellets; food direction inputs; prey earns .25 per pellet.',
            'hunger': 'Energy declines .06/second; maximum speed scales from 100% to 30% as energy declines. No starvation.',
            'joints': 'Turning-limited bodies with two fixed .65-unit capsule centerlines and an actuated bend.',
            'swimming': 'Jointed bodies propelled by squared bend speed with exponential drag; speed-dependent steering.',
        },
        'note':'Simplified standalone mechanics; joints vs turning; swimming vs joints; other additions vs baseline. Fixed reference opponents are within-condition.'},indent=2),encoding='utf-8')


def plot(out, rows, timings, final_generation):
    fig, axes = plt.subplots(1,3,figsize=(14,5.2),layout='constrained')
    names = ['Empty arena','Faster predator','Turning limits','Acceleration','Obstacles','Food','Hunger','Jointed body','Tail propulsion']
    results = {}
    for condition in CONDITIONS:
        parent = 'joints' if condition=='swimming' else ('turning' if condition=='joints' else 'baseline')
        def values(c,g,key):
            return np.array([r[key] for r in rows if r['condition']==c and r['generation']==g])
        gains=[]
        for metric in ('predator_capture','prey_escape'):
            learning=values(condition,final_generation,metric)-values(condition,0,metric)
            reference=values(parent,final_generation,metric)-values(parent,0,metric)
            gains.append((learning-reference)*100)
        results[condition]=gains
    for index,(metric,title) in enumerate([('predator_capture','Effect on predator learning'),('prey_escape','Effect on prey learning')]):
        ax=axes[index]
        for i,condition in enumerate(CONDITIONS):
            v=results[condition][index]
            ax.barh(i,v.mean(),color='#3f789f' if index==0 else '#c47a46',alpha=.65)
            ax.scatter(v,np.arange(len(v))*.035+i-.07,color='#162a39',s=14,zorder=3)
        ax.axvline(0,color='#777',lw=1)
        ax.set(yticks=range(len(names)),yticklabels=names if index==0 else [],xlabel='Change in learning gain (percentage points)',title=title)
        ax.invert_yaxis()
    ax=axes[2]
    means=[np.mean([t['seconds'] for t in timings if t['condition']==c]) for c in CONDITIONS]
    ax.barh(range(len(names)),np.array(means)/means[0],color='#5c9786')
    ax.set(yticks=range(len(names)),yticklabels=[],xlabel='Runtime relative to empty arena',title='Computational cost')
    ax.invert_yaxis()
    fig.suptitle('Adding one mechanic at a time: matched co-evolution pilot',fontweight='bold',fontsize=15)
    fig.text(.5,-.015,'Dots show seeds. Joints are compared with turning limits; swimming with joints; other additions with the empty arena.',ha='center',fontsize=9)
    fig.savefig(out/'element-effects.png',dpi=170,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for condition,name in zip(CONDITIONS,names):
        for ax,key in zip(axes,('predator_capture','prey_escape')):
            gens=sorted(set(r['generation'] for r in rows if r['condition']==condition))
            means=[np.mean([r[key] for r in rows if r['condition']==condition and r['generation']==g])*100 for g in gens]
            ax.plot(gens,means,'o-',label=name,markersize=3)
    axes[0].set(title='Capture against initial prey',xlabel='Generations',ylabel='Capture rate (%)')
    axes[1].set(title='Escape against initial predator',xlabel='Generations',ylabel='Escape rate (%)')
    axes[1].legend(fontsize=8,loc='center left',bbox_to_anchor=(1, .5))
    fig.savefig(out/'learning-curves.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    main()
