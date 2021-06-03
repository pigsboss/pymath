"""Spherical functions.
"""

## python 2 and 3 compatible.
##

from pymath.common import *
import pymath.quaternion as quat

def gnomonic(pt,pu):
    '''Gnomonic projection is a mapping from unit sphere to its tangent plane.

Given a point pu on the unit sphere and the tangent plane at point pt, the
point pt is projected along from the unit sphere to the plane by casting a
ray from the sphere centre to the plane through pt.
pp is the point on the tangent plane.
Points are specified in cartesian coordinates.
'''
    if np.size(pt) != np.size(pu):
        v = np.sum(np.multiply(pt.T,pu.T).T,axis=0)
    else:
        v = np.sum(np.multiply(pt,pu),axis=0)
    if np.any(v <= DEPS):
        if np.size(v) > 1:
            pp = np.zeros((3, np.size(v)))
            pp[:,v>DEPS] = pu[:,v>DEPS] / v[v>DEPS]
            _, m = np.broadcast_arrays(pp,v)
            return np.ma.masked_array(pp,mask=(m<=DEPS))
        else:
            raise StandardError('No point can be projected to the plane.')
    else:
        return np.divide(pu, v)

def geodesic(a,b):
    """Geodesic angle and distance from the point a to point b.

The geodesic angle is the angle from the meridian passing point a to the
geodesic from a to b.
In our convention a meridian goes from the south pole to the north pole.
Both a and b are point(s) on the unit sphere.
a or b can be given in the following ways:
  1, a is a point while b is another point.
  2, one of them is a point while the other is an array of points.
  3, both of them are arrays of points, but in this way the two arrays
  must have the same numbers of points.
Points on the spherical surface can be given in either their Cartesian
coordinates as (x,y,z).T or in polar angles as (phi,theta).T
"""
    adim = np.shape(a)
    if adim[0] == 3:
        phi,theta,_ = xyz2ptr(a[0],a[1],a[2])
    elif adim[0] == 2:
        phi,theta = a
    else:
        raise StandardError('column vectors of a must be either'\
            + ' (phi,theta) or (x,y,z).')
    q = quat.from_angles(phi-np.pi, np.pi*0.5-theta)
    p = quat.rotate(quat.conjugate(q), as_xyz(b))
    phi,theta,_ = xyz2ptr(p[0],p[1],p[2])
    return phi,np.pi*0.5-theta

def angle(a,b):
    """Geodesic angle and distance from the point a to point b.

The geodesic angle is the angle from the meridian passing point a to the
geodesic from a to b.
In our convention a meridian goes from the south pole to the north pole.
Both a and b are point(s) on the unit sphere.
a or b can be given in the following ways:
  1, a is a point while b is another point.
  2, one of them is a point while the other is an array of points.
  3, both of them are arrays of points, but in this way the two arrays
  must have the same numbers of points.
Points on the spherical surface can be given in either their Cartesian
coordinates as (x,y,z).T or in polar angles as (phi,theta).T
"""
    return geodesic(a,b)[0]

def distance(a,b):
    """Geodesic distance between a and b.

Both a and b are point(s) on the unit spherical surface.
a or b can be given in the following ways:
  1, a is a point while b is another point.
  2, one of them is a point while the other is an array of points.
  3, both of them are arrays of points, but in this way the two arrays must
     have the same numbers of points.
Points on the spherical surface can be given in either their Cartesian
coordinates as (x,y,z).T or in polar angles as (phi,theta).T
"""
    # convert polar-angles vectors to unit x-y-z vectors
    a = direction(as_xyz(a))
    b = direction(as_xyz(b))
    return np.arccos(np.sum(np.multiply(a.T,b.T).T,axis=0))

def arc(a,b,npts=10):
    """Points of arc on geodesic between a and b.

"""
    a = direction(as_xyz(a))
    b = direction(as_xyz(b))
    d = distance(a,b)
    pgv = np.array(range(0,npts)) / np.double(npts - 1) * d
    r = np.array([np.cos(pgv),np.sin(pgv),np.zeros(npts)])
    up = direction(np.cross(a,b))
    r = quat.rotate(quat.from_axes(a,up), r)
    return r[0],r[1],r[2]

def nearest(Phi,Theta,phi,theta):
    """Find the geodesic nearest location on the grid defined by Phi and Theta, given arbitrary point(s) (phi,theta).

"""
    N = np.double(Phi.shape)
    r = [0] * Phi.ndim
    while (N > 1).any():
        r_sub,n,idx_q = subarray(r=r,N=N)
        q = np.zeros([2,len(idx_q)])
        for i in range(0,len(idx_q)):
            q[0,i] = Phi[idx_q[i]]
            q[1,i] = Theta[idx_q[i]]
        d = distance(q,np.array([phi,theta]))
        idx = list(d).index(min(d))
        r = np.array(r_sub[idx])
        N = np.array(n)
    return idx_q[idx],d[idx]

def rotation_axis(a,b):
    """Find rotation axis.

c = geodesic_axis(a,b)
a and b must be unit x-y-z vectors.
c is the rotation axis of a and b.
c = (a X b) / ||a|| / ||b||
"""
    c = np.cross(a,b)
    d = norm(c)
    if d <= DEPS:
        phi,theta,rho = xyz2ptr(a[0],a[1],a[2])
        c = quat.rotate(quat.from_angles(phi,theta,0.0), \
            [0.0, 0.0, 1.0])
    else:
        c = c/d
    return c

def intersection(a,b,c,d):
    """Geodesic intersections of arc(a,b) and arc(c,d).

a, b, c, and d are points on unit sphere.
Each of them can be either a single point or an array of points.
"""
    a = as_xyz(a)
    b = as_xyz(b)
    c = as_xyz(c)
    d = as_xyz(d)
    norm_ab = rotation_axis(a,b)
    norm_cd = rotation_axis(c,d)
    return rotation_axis(norm_ab,norm_cd)
