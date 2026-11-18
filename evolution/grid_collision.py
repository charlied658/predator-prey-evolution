"""Swept circular bodies against the solid cells of an exported landscape."""
import math


class GridCollider:
    def __init__(self, landscape, radius=2.4):
        self.size = landscape['resolution']
        self.solid = landscape['solid']
        self.radius = radius

    def cells(self, start, end, radius):
        n = self.size
        for y in range(max(0, math.floor(min(start[1], end[1]) - radius)),
                       min(n, math.floor(max(start[1], end[1]) + radius) + 1)):
            for x in range(max(0, math.floor(min(start[0], end[0]) - radius)),
                           min(n, math.floor(max(start[0], end[0]) + radius) + 1)):
                if self.solid[y*n+x]:
                    yield x, y

    def clear(self, position):
        r, n = self.radius, self.size
        x, y = position
        if min(x, y) < r or max(x, y) > n-r:
            return False
        for cx, cy in self.cells(position, position, r):
            nearest_x = max(cx, min(cx+1, x))
            nearest_y = max(cy, min(cy+1, y))
            if (nearest_x-x)**2 + (nearest_y-y)**2 < r*r - 1e-9:
                return False
        return True

    def sweep(self, start, motion):
        r, n = self.radius, self.size
        end = (start[0]+motion[0], start[1]+motion[1])
        best, normal = 1.0, None

        def accept(t, nx, ny):
            nonlocal best, normal
            if -1e-10 <= t <= best and motion[0]*nx + motion[1]*ny < -1e-10:
                best, normal = max(0, t), (nx, ny)

        for axis in (0, 1):
            delta = motion[axis]
            if abs(delta) < 1e-12:
                continue
            boundary = n-r if delta > 0 else r
            t = (boundary-start[axis])/delta
            sign = -1 if delta > 0 else 1
            accept(t, sign if axis==0 else 0, sign if axis==1 else 0)
        length_squared = motion[0]**2 + motion[1]**2
        if length_squared < 1e-18:
            return best, normal
        for x, y in self.cells(start, end, r):
            for axis, lower, upper, cross_lower in ((0, x-r, x+1+r, y), (1, y-r, y+1+r, x)):
                delta = motion[axis]
                if abs(delta) < 1e-12:
                    continue
                t = ((lower if delta > 0 else upper) - start[axis])/delta
                cross = start[1-axis] + t*motion[1-axis]
                if cross_lower <= cross <= cross_lower+1:
                    sign = -1 if delta > 0 else 1
                    accept(t, sign if axis==0 else 0, sign if axis==1 else 0)
            for cx, cy in ((x,y), (x+1,y), (x,y+1), (x+1,y+1)):
                ox, oy = start[0]-cx, start[1]-cy
                dot = ox*motion[0]+oy*motion[1]
                discriminant = dot*dot-length_squared*(ox*ox+oy*oy-r*r)
                if dot < 0 and discriminant >= 0:
                    t = (-dot-math.sqrt(discriminant))/length_squared
                    nx, ny = ox+t*motion[0], oy+t*motion[1]
                    length = math.hypot(nx, ny)
                    if length > 1e-12:
                        accept(t, nx/length, ny/length)
        return best, normal

    def move(self, position, displacement):
        if not self.clear(position):
            raise ValueError('Body starts overlapping terrain')
        current = tuple(position)
        remaining = tuple(displacement)
        contacts = 0
        for _ in range(6):
            distance = math.hypot(*remaining)
            if distance < 1e-9:
                break
            fraction, normal = self.sweep(current, remaining)
            travel = fraction if normal is None else max(0, fraction-1e-6/distance)
            current = (current[0]+remaining[0]*travel, current[1]+remaining[1]*travel)
            if normal is None:
                break
            contacts += 1
            residual = (remaining[0]*(1-fraction), remaining[1]*(1-fraction))
            inward = residual[0]*normal[0]+residual[1]*normal[1]
            remaining = (residual[0]-inward*normal[0], residual[1]-inward*normal[1])
        if not self.clear(current):
            raise RuntimeError('Collision solver left a body inside terrain')
        return current, contacts
