"""Longer replays of saved hybrid controllers; no evolutionary training."""
from src.analysis.replay_fonts import replay_font
import argparse,base64,io,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.simulation.single_pellet import evaluate,hybrid_exploration_sites,HYBRID_INSTRUCTIONS
from src.simulation.vision_pilot import setup,DT


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--seconds',type=float,default=120);p.add_argument('--enable-recovery',action='store_true');p.add_argument('--output',type=Path,default=Path('results/hybrid-long-replays'));args=p.parse_args()
    if args.seconds<=0:p.error('Duration must be positive')
    args.output.mkdir(parents=True,exist_ok=False)
    saved=json.loads((args.run/'instructions.json').read_text())
    for key in ('steering_limit_degrees','minimum_throttle','goal_reached_distance'):
        if saved[key]!=HYBRID_INSTRUCTIONS[key]:raise ValueError('Saved movement instructions differ from current configuration')
    design=json.loads((args.run/'design.json').read_text());count=design['pellets'];grid=design['grid_cells']
    terrain=json.loads((args.run/'terrain.json').read_text());world=setup(terrain);_,eligible=hybrid_exploration_sites(world,grid);denominator=int(eligible.sum())
    champions=json.loads((args.run/f'{count}-pellets'/'champions.json').read_text());examples=[];summary=[]
    for seed in range(min(3,len(champions))):
        start=90000000+seed*1000;trace=[]
        result=evaluate(np.array(champions[seed])[None],[start],world,trace=trace,pellet_count=count,food_layout='scattered',grid_cells=grid,controller='hybrid',duration_seconds=args.seconds,recovery=args.enable_recovery or saved.get('recovery_enabled',False))
        pos=np.array([f['position'] for f in trace]);stall=[]
        for i,frame in enumerate(trace):
            window=pos[max(0,i-20):i+1]
            stationary=i>=20 and np.linalg.norm(window-window[-1],axis=1).max()<1 and not frame['collected']
            frame['stalled']=bool(stationary);frame['coverage']=sum(eligible[j] for j in frame['visited'])/denominator
            stall.append(stationary)
        changes=[i*DT for i in range(1,len(trace)) if trace[i]['count']>trace[i-1]['count']]
        stalled_indices=np.flatnonzero(stall)
        item=dict(seed=seed+1,start=start,food=int(result[0,2]),pellets=count,coverage_percent=100*trace[-1]['coverage'],visited_tiles=len(trace[-1]['visited']),accessible_tiles=denominator,simulated_seconds=float(result[0,1]),wall_contacts=trace[-1]['contacts'],wall_contact_steps=sum(f['last_contacts']>0 for f in trace[1:]),stalled_seconds=float(np.sum(stall)*DT),first_stall_seconds=float(stalled_indices[0]*DT) if len(stalled_indices) else None,last_food_seconds=changes[-1] if changes else None,distance_travelled=float(np.linalg.norm(np.diff(pos,axis=0),axis=1).sum()),recoveries=trace[-1].get('recoveries',0))
        examples.append(dict(seed=seed+1,start=start,frames=trace));summary.append(item);print(json.dumps(item),flush=True)
    applied=dict(HYBRID_INSTRUCTIONS,recovery_enabled=args.enable_recovery or saved.get('recovery_enabled',False))
    (args.output/'applied-instructions.json').write_text(json.dumps(applied,indent=2))
    payload=dict(seconds=args.seconds,dt=DT,pellets=count,grid_cells=grid,accessible_tiles=denominator,terrain=terrain,examples=examples)
    (args.output/'replays.json').write_text(json.dumps(payload));(args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    font=replay_font(15);small=replay_font(14)
    tile,gap,top=320,16,70;scale=tile/96;solid=np.array(terrain['solid']).reshape(96,96)
    bg=Image.fromarray(np.where(solid[:,:,None],np.array([108,119,116]),np.array([232,228,206])).astype('uint8')).resize((tile,tile),Image.Resampling.NEAREST).convert('RGBA')
    frames=[]
    for tick in range(0,int(round(args.seconds/DT))+1,4):
        canvas=Image.new('RGB',(len(examples)*tile+(len(examples)+1)*gap,top+tile+100),'#172321');d=ImageDraw.Draw(canvas)
        d.text((gap,7),f'Long terrain replays | {args.seconds:g} simulated seconds | playback 4x | NO retraining',font=font,fill='white')
        d.text((gap,30),'Green: visited | purple: target | red body outline: stalled for 2s | circle: vision',font=small,fill='#d6e4dc')
        for panel,example in enumerate(examples):
            trace=example['frames'];i=min(tick,len(trace)-1);f=trace[i];x=gap+panel*(tile+gap);image=bg.copy()
            overlay=np.zeros((96,96,4),dtype='uint8');size=96//grid
            for cell in f['visited']:
                gx,gy=cell%grid,cell//grid;overlay[gy*size:(gy+1)*size,gx*size:(gx+1)*size]=[50,160,85,90]
            overlay[solid.astype(bool),3]=0;image=Image.alpha_composite(image,Image.fromarray(overlay).resize((tile,tile),Image.Resampling.NEAREST));pd=ImageDraw.Draw(image)
            for line in range(grid+1):
                q=min(tile-1,round(line*tile/grid));pd.line((q,0,q,tile-1),fill='#9da995');pd.line((0,q,tile-1,q),fill='#9da995')
            points=[tuple(np.array(fr['position'])*scale) for fr in trace[max(0,i-100):i+1]]
            if len(points)>1:pd.line(points,fill='#77a7b0',width=2)
            px,py=np.array(f['position'])*scale;r=24*scale;pd.ellipse((px-r,py-r,px+r,py+r),outline='#66a7a6')
            if f.get('target') is not None:
                tx,ty=np.array(f['target'])*scale;pd.line((px,py,tx,ty),fill='#9a58ac');pd.ellipse((tx-4,ty-4,tx+4,ty+4),outline='#9a58ac',width=2)
            for point in f['remaining']:
                fx,fy=np.array(point)*scale;pd.ellipse((fx-3,fy-3,fx+3,fy+3),fill='#e3b72f',outline='#66551a')
            r=2.4*scale;pd.ellipse((px-r,py-r,px+r,py+r),fill='#239b9d',outline='#e45c50' if f['stalled'] else 'white',width=3)
            canvas.paste(image.convert('RGB'),(x,top));d.text((x,top-20),f'Seed {example["seed"]} | start {example["start"]}',font=small,fill='white')
            lines=[f'{tick*DT:.1f}s | food {f["count"]}/{count} | {f.get("mode","")}',f'Coverage {f["coverage"]:.1%} ({len(f["visited"])}/{denominator} tiles)',f'Wall contacts {f["contacts"]} | '+('STALLED' if f['stalled'] else 'moving / complete')]
            for line,text in enumerate(lines):d.text((x,top+tile+8+line*23),text,font=small,fill='#ef8f83' if line==2 and f['stalled'] else '#d6e4dc')
        frames.append(canvas)
    frames[0].save(args.output/'long-replays.gif',save_all=True,append_images=frames[1:],duration=100,loop=0);frames[min(75,len(frames)-1)].save(args.output/'preview.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for example in examples:
        trace=example['frames'];times=np.arange(len(trace))*DT
        axes[0].plot(times,[f['count'] for f in trace],label=f'Seed {example["seed"]}');axes[1].plot(times,[100*f['coverage'] for f in trace])
    axes[0].set(title='Pellets collected',xlabel='Simulated seconds',ylabel='Pellets',ylim=(0,count));axes[0].legend();axes[1].set(title='Reachable grid tiles entered',xlabel='Simulated seconds',ylabel='Coverage (%)',ylim=(0,100));fig.savefig(args.output/'long-run-progress.png',dpi=170);plt.close(fig)
    # A standalone viewer permits real-time playback, pausing and scrubbing.
    rgba=np.zeros((96,96,4),dtype='uint8');rgba[solid.astype(bool)]=[108,119,116,255];buffer=io.BytesIO();Image.fromarray(rgba).save(buffer,format='PNG');rocks=base64.b64encode(buffer.getvalue()).decode()
    html='''<!doctype html><meta charset="utf-8"><title>Long hybrid replays</title><style>body{background:#172321;color:#edf0e7;font:16px system-ui;margin:20px}canvas{max-width:100%;background:#172321}input{width:45%}button,select{font:inherit;margin:8px;padding:5px}p{max-width:950px}</style><h2>Long terrain replays</h2><p>Green = visited; cream = unvisited; purple = current target; red outline = stalled. Coverage counts reachable grid tiles entered by the body centre, not all visible area.</p><button id="play">Pause</button><select id="rate"><option value="1">1x</option><option value="2">2x</option><option value="4" selected>4x</option><option value="8">8x</option></select><input id="seek" type="range" min="0" step="0.1"><span id="time"></span><br><canvas id="view" width="1040" height="460"></canvas><script>const data=PAYLOAD,rocks=new Image();rocks.src='data:image/png;base64,ROCKS';const c=document.getElementById('view'),ctx=c.getContext('2d'),slider=document.getElementById('seek');slider.max=data.seconds;let t=0,playing=true,last=performance.now();document.getElementById('play').onclick=()=>{playing=!playing;document.getElementById('play').textContent=playing?'Pause':'Play'};slider.oninput=()=>{t=Number(slider.value)};function draw(){ctx.clearRect(0,0,c.width,c.height);for(let k=0;k<data.examples.length;k++){let e=data.examples[k],i=Math.min(Math.round(t/data.dt),e.frames.length-1),f=e.frames[i],x=16+k*336,y=30,s=320/96;ctx.save();ctx.fillStyle='#e8e4ce';ctx.fillRect(x,y,320,320);ctx.fillStyle='rgba(50,160,85,.35)';for(let cell of f.visited){ctx.fillRect(x+(cell%data.grid_cells)*320/data.grid_cells,y+Math.floor(cell/data.grid_cells)*320/data.grid_cells,320/data.grid_cells,320/data.grid_cells)}ctx.drawImage(rocks,x,y,320,320);ctx.strokeStyle='#9da995';for(let g=0;g<=data.grid_cells;g++){let q=g*320/data.grid_cells;ctx.beginPath();ctx.moveTo(x+q,y);ctx.lineTo(x+q,y+320);ctx.moveTo(x,y+q);ctx.lineTo(x+320,y+q);ctx.stroke()}ctx.beginPath();ctx.strokeStyle='#77a7b0';for(let j=Math.max(0,i-100);j<=i;j++){let p=e.frames[j].position;j===Math.max(0,i-100)?ctx.moveTo(x+p[0]*s,y+p[1]*s):ctx.lineTo(x+p[0]*s,y+p[1]*s)}ctx.stroke();let px=x+f.position[0]*s,py=y+f.position[1]*s;ctx.strokeStyle='#66a7a6';ctx.beginPath();ctx.arc(px,py,24*s,0,7);ctx.stroke();if(f.target){ctx.strokeStyle='#9a58ac';ctx.beginPath();ctx.moveTo(px,py);ctx.lineTo(x+f.target[0]*s,y+f.target[1]*s);ctx.stroke()}ctx.fillStyle='#e3b72f';for(let p of f.remaining){ctx.beginPath();ctx.arc(x+p[0]*s,y+p[1]*s,3,0,7);ctx.fill()}ctx.fillStyle='#239b9d';ctx.strokeStyle=f.stalled?'#e45c50':'white';ctx.lineWidth=3;ctx.beginPath();ctx.arc(px,py,2.4*s,0,7);ctx.fill();ctx.stroke();ctx.fillStyle='white';ctx.font='14px system-ui';ctx.fillText('Seed '+e.seed+' | start '+e.start,x,20);ctx.fillText('Food '+f.count+'/'+data.pellets+' | '+(f.mode||''),x,375);ctx.fillText('Coverage '+(100*f.coverage).toFixed(1)+'% | '+f.visited.length+'/'+data.accessible_tiles+' tiles',x,400);ctx.fillStyle=f.stalled?'#ef8f83':'white';ctx.fillText('Wall contacts '+f.contacts+' | '+(f.stalled?'STALLED':'moving / complete'),x,425);ctx.restore()}document.getElementById('time').textContent=t.toFixed(1)+'s / '+data.seconds+'s'}function loop(now){if(playing)t=Math.min(data.seconds,t+(now-last)/1000*Number(document.getElementById('rate').value));last=now;slider.value=t;draw();requestAnimationFrame(loop)}rocks.onload=()=>requestAnimationFrame(loop);</script>'''
    (args.output/'viewer.html').write_text(html.replace('PAYLOAD',json.dumps(payload)).replace('ROCKS',rocks),encoding='utf-8')
    (args.output/'design.json').write_text(json.dumps(dict(source=str(args.run),seconds=args.seconds,selection='First three controller seeds; first held-out start per seed; not selected for success',coverage='Fraction of body-accessible grid tiles entered by the centre; not observed ground area',stall='Body remained within 1 map cell of its current location for the previous 2 seconds, while food remains. Does not by itself prove obstacle contact caused the stall.',wall_contacts='Counts solver contact iterations, not unique obstacles',playback='GIF 4x; standalone viewer has 1x/2x/4x/8x and seek controls',recovery_enabled=args.enable_recovery or saved.get('recovery_enabled',False),training='Saved controllers; no retraining. Recovery, if explicitly enabled, is an intervention on the programmed rules'),indent=2))

if __name__=='__main__':main()
