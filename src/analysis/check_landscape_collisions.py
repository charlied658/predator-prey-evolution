"""Record terrain collision demonstrations and verify them independently."""
from src.analysis.replay_fonts import replay_font
import argparse
from collections import deque
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.terrain.grid_collision import GridCollider


def route(collider, start, rng):
    n = collider.size
    origin = (int(start[0]), int(start[1]))
    queue = deque([origin])
    parents = {origin: None}
    while queue:
        x,y = queue.popleft()
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            cell = (x+dx,y+dy)
            center = (cell[0]+.5,cell[1]+.5)
            if cell in parents or not (0<=cell[0]<n and 0<=cell[1]<n):
                continue
            if collider.clear(center):
                parents[cell] = (x,y)
                queue.append(cell)
    choices = [cell for cell in parents if math.dist(origin,cell)>15]
    target = rng.choice(choices or list(parents))
    path = []
    while target is not None:
        path.append((target[0]+.5,target[1]+.5))
        target = parents[target]
    return deque(reversed(path))


def record(landscape):
    rng = random.Random(42)
    collider = GridCollider(landscape)
    starts = [(x+.5,y+.5) for x,y in landscape['markers'][:2]]
    positions = starts.copy()
    path = route(collider, positions[1], rng)
    walls = [(i%collider.size+.5,i//collider.size+.5)
             for i,solid in enumerate(collider.solid) if solid]
    target = rng.choice(walls)
    frames, contact_total = [], [0,0]
    blocked = np.array([(i%collider.size,i//collider.size)
                        for i,solid in enumerate(collider.solid) if solid])
    for tick in range(601):
        if tick:
            if tick%90==1:
                target = rng.choice(walls)
            if math.dist(positions[1],path[0]) < .25:
                path.popleft()
                if not path:
                    path = route(collider,positions[1],rng)
            commands = []
            for side,(goal,speed) in enumerate(((target,42),(path[0],14))):
                delta = (goal[0]-positions[side][0],goal[1]-positions[side][1])
                distance = max(math.hypot(*delta),1e-9)
                step = min(distance,speed/60)
                commands.append((delta[0]/distance*step,delta[1]/distance*step))
            for side,command in enumerate(commands):
                positions[side],contacts = collider.move(positions[side],command)
                contact_total[side] += contacts
                # Validate against every cell, independently of the sweep's
                # bounding-box filtering and contact calculations.
                nearest = np.maximum(blocked,np.minimum(blocked+1,positions[side]))
                distance_squared = np.sum((nearest-positions[side])**2,axis=1)
                if distance_squared.min() < collider.radius**2-1e-8:
                    raise AssertionError('Recorded body penetrated terrain')
        frames.append({'positions':positions.copy(),'contacts':contact_total.copy()})
    return {'map':landscape,'radius':collider.radius,'fps':60,'frames':frames}


def animate(recordings, output):
    tile, gutter, top = 384, 18, 64
    width = 3*tile+4*gutter
    font = replay_font(19)
    small = replay_font(14)
    backgrounds=[]
    for recording in recordings:
        data=recording['map']
        solid=np.array(data['solid']).reshape(data['resolution'],data['resolution'])
        pixels=np.where(solid[:,:,None],np.array([108,119,116]),np.array([232,228,206])).astype('uint8')
        backgrounds.append(Image.fromarray(pixels).resize((tile,tile),Image.Resampling.NEAREST))
    frames=[]
    for tick in range(0,601,4):
        canvas=Image.new('RGB',(width,top+tile+62),'#172321')
        draw=ImageDraw.Draw(canvas)
        draw.text((gutter,9),'Collision replay: coral pushes into terrain; teal follows a clearance-aware path',font=font,fill='white')
        for i,(recording,bg) in enumerate(zip(recordings,backgrounds)):
            x=gutter+i*(tile+gutter)
            canvas.paste(bg,(x,top))
            draw.text((x,39),recording['map']['preset'],font=small,fill='#d6e4dc')
            scale=tile/recording['map']['resolution']
            for side,color in enumerate(('#e1755e','#239b9d')):
                points=[(x+f['positions'][side][0]*scale,top+f['positions'][side][1]*scale)
                        for f in recording['frames'][max(0,tick-100):tick+1:4]]
                if len(points)>1:
                    draw.line(points,fill=color,width=2)
                px,py=recording['frames'][tick]['positions'][side]
                px,py=x+px*scale,top+py*scale
                radius=recording['radius']*scale
                draw.ellipse((px-radius,py-radius,px+radius,py+radius),fill=color,outline='white',width=1)
            count=recording['frames'][tick]['contacts']
            draw.text((x,top+tile+9),f'{tick/60:.1f}s  |  contacts {count[0]} / {count[1]}',font=small,fill='#d6e4dc')
        frames.append(canvas)
    frames[0].save(output,save_all=True,append_images=frames[1:],duration=67,loop=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/landscape-collisions'))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    recordings=[]
    for i in range(3):
        landscape=json.loads(Path(f'results/landscape-previews/map-{i}-seed-42.json').read_text())
        result=record(landscape)
        recordings.append(result)
        (args.output/f'replay-{i}.json').write_text(json.dumps(result),encoding='utf-8')
    animate(recordings,args.output/'collision-replays.gif')
    summary={'recorded_frames_per_map':601,'maps':3,'agents_per_map':2,'penetrations':0,
             'contacts':[r['frames'][-1]['contacts'] for r in recordings],
             'note':'Scripted navigation and wall-pressure tests; agents do not collide with each other.'}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary))


if __name__=='__main__':
    main()
