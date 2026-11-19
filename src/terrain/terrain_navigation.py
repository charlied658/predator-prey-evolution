"""Static terrain knowledge: body-clear routes and point line-of-sight sensors."""
from collections import OrderedDict
import math
import numpy as np
from src.terrain.grid_collision import GridCollider


class TerrainNavigation:
    def __init__(self, collider, sites):
        self.collider = collider
        self.sight = GridCollider(dict(resolution=collider.size, solid=collider.solid), radius=0)
        self.sites = np.asarray(sites)
        self.solid = np.asarray(collider.solid,dtype=bool).reshape(collider.size,collider.size)
        lookup = {tuple(p): i for i, p in enumerate(self.sites)}
        self.edges = [[] for _ in sites]
        for i, p in enumerate(self.sites):
            for offset in ((1,0),(0,1)):
                j = lookup.get(tuple(p+offset))
                if j is not None and self.clear_segment(p,self.sites[j]):
                    self.edges[i].append(j)
                    self.edges[j].append(i)
        self.fields = OrderedDict()

    def clear_segment(self, a, b):
        return self.segment_clear(a,b,self.collider.radius)

    def segment_clear(self, a, b, radius):
        """Boolean version of swept-circle collision; no contact normal needed.

        Vectorize only occupied cells in the swept bounding box. Test the same
        faces and rounded corners as GridCollider.sweep, with its tolerances.
        """
        ax,ay=map(float,a);bx,by=map(float,b);dx,dy=bx-ax,by-ay
        size=self.collider.size
        for origin,delta in ((ax,dx),(ay,dy)):
            if abs(delta)>=1e-12:
                t=((size-radius if delta>0 else radius)-origin)/delta
                if -1e-10<=t<=1 and abs(delta)>1e-10:return False
        length=dx*dx+dy*dy
        if length<1e-18:return True
        x0=max(0,math.floor(min(ax,bx)-radius));x1=min(size,math.floor(max(ax,bx)+radius)+1)
        y0=max(0,math.floor(min(ay,by)-radius));y1=min(size,math.floor(max(ay,by)+radius)+1)
        yy,xx=np.nonzero(self.solid[y0:y1,x0:x1])
        if not len(xx):return True
        x=xx+x0;y=yy+y0
        if abs(dx)>1e-10:
            t=((x-radius if dx>0 else x+1+radius)-ax)/dx
            cross=ay+t*dy
            if np.any((t>=-1e-10)&(t<=1)&(cross>=y)&(cross<=y+1)):return False
        if abs(dy)>1e-10:
            t=((y-radius if dy>0 else y+1+radius)-ay)/dy
            cross=ax+t*dx
            if np.any((t>=-1e-10)&(t<=1)&(cross>=x)&(cross<=x+1)):return False
        if radius==0:return True
        for ox,oy in ((0,0),(1,0),(0,1),(1,1)):
            rx=ax-x-ox;ry=ay-y-oy;dot=rx*dx+ry*dy
            disc=dot*dot-length*(rx*rx+ry*ry-radius*radius)
            valid=(dot<0)&(disc>=0)
            t=(-dot-np.sqrt(np.maximum(disc,0)))/length
            # At the first root, motion dot outward normal = -sqrt(disc)/radius.
            if np.any(valid&(t>=-1e-10)&(t<=1)&(disc>(1e-10*radius)**2)):return False
        return True

    def visible(self, a, b, radius):
        delta=np.asarray(b)-a
        return bool(delta@delta<=radius*radius and self.segment_clear(a,b,0))

    def anchor(self, p):
        distances=np.sum((self.sites-p)**2,axis=1)
        for i in np.argsort(distances):
            if self.clear_segment(p,self.sites[i]):return int(i)
        return None

    def field(self, target):
        if target in self.fields:
            self.fields.move_to_end(target)
            return self.fields[target]
        # Every edge costs one: layered BFS avoids the heap. Sorting each layer
        # preserves the old Dijkstra node-index tie breaking and exact routes.
        distance=[float('inf')]*len(self.sites);distance[target]=0
        parent=[-1]*len(self.sites);layer=[target];cost=0
        while layer:
            following=[]
            for i in sorted(layer):
                for j in self.edges[i]:
                    if cost+1<distance[j]:
                        distance[j]=cost+1;parent[j]=i;following.append(j)
            layer=following;cost+=1
        distance=np.asarray(distance);parent=np.asarray(parent)
        self.fields[target]=(distance,parent)
        if len(self.fields)>128:self.fields.popitem(last=False)
        return distance,parent

    def waypoint(self, position, goal):
        if self.clear_segment(position,goal):return np.asarray(goal).copy()
        start,end=self.anchor(position),self.anchor(goal)
        if start is None or end is None:return np.asarray(position).copy()
        distance,parent=self.field(end)
        if not np.isfinite(distance[start]):return np.asarray(position).copy()
        path=[start]
        while path[-1]!=end:path.append(int(parent[path[-1]]))
        # Smooth the grid route by selecting the furthest directly clear point.
        for i in reversed(path):
            if self.clear_segment(position,self.sites[i]):return self.sites[i].copy()
        return np.asarray(position).copy()

    def vision_outline(self, position, radius, rays=64):
        points=[]
        for angle in np.arange(rays)*2*np.pi/rays:
            delta=radius*np.array([np.cos(angle),np.sin(angle)])
            fraction,_=self.sight.sweep(position,delta)
            points.append(np.asarray(position)+delta*fraction)
        return points


def navigation(world):
    """Share immutable terrain graph/cache across encounters, never visit memory."""
    if not hasattr(world[0],'navigation'):
        world[0].navigation=TerrainNavigation(world[0],world[1])
    return world[0].navigation
