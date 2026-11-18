"""Seeded, tunable multiscale value-noise terrain for the 96-cell simulator."""
from collections import deque
import numpy as np
from PIL import Image


def generate_terrain(seed=42,openness=.7,roughness=.35,feature_size=16.,kind='Outcrops'):
    if not .35<=openness<=.95 or not 0<=roughness<=1 or not 6<=feature_size<=32:
        raise ValueError('Use openness 0.35–0.95, roughness 0–1, feature size 6–32')
    if kind not in ('Outcrops','Ridges','Clearings'):raise ValueError('Unknown terrain type')
    if not 0<=int(seed)<2**63:raise ValueError('Map seed must be a nonnegative integer below 2^63')
    rng=np.random.default_rng(int(seed));n=96;field=np.zeros((n,n))
    for octave,amplitude in enumerate((1.,.08+.5*roughness,.02+.25*roughness)):
        frequency=2**octave*n/feature_size
        nx,ny=(frequency*.5,frequency*1.5) if kind=='Ridges' else (frequency,frequency)
        if kind=='Clearings':nx*=.7;ny*=.7
        noise=rng.normal(size=(max(3,round(ny)+2),max(3,round(nx)+2))).astype('float32')
        layer=np.asarray(Image.fromarray(noise).resize((n,n),Image.Resampling.BICUBIC),float)
        field+=amplitude*layer
    solid=field>=np.quantile(field,openness)
    # Preserve the largest connected open region, as exported maps do.
    unseen=~solid.copy();largest=[]
    for y,x in np.argwhere(unseen):
        if not unseen[y,x]:continue
        component=[(int(y),int(x))];unseen[y,x]=False;q=deque(component)
        while q:
            cy,cx=q.popleft()
            for yy,xx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                if 0<=yy<n and 0<=xx<n and unseen[yy,xx]:
                    unseen[yy,xx]=False;component.append((yy,xx));q.append((yy,xx))
        if len(component)>len(largest):largest=component
    solid[:]=True
    if largest:
        yy,xx=np.array(largest).T;solid[yy,xx]=False
    return dict(resolution=n,solid=solid.astype(int).ravel().tolist(),preset=kind,seed=int(seed),generator='multiscale-value-noise-v1',requested_openness=float(openness),open_fraction=float((~solid).mean()),roughness=float(roughness),feature_size=float(feature_size))
