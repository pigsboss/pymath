"""Common math functions.
"""

## python 2 and 3 compatible.
##

import numpy as np
import numexpr as ne
from mpmath import mp
DEPS = np.finfo(np.double).eps

def print_RK4tableau_Ralston(d):
    mp.dps=d
    print("a2 = 0.4")
    print("a3 ={}".format((   14-   3*mp.sqrt(5))/  16))
    print("b31={}".format((-2889+1428*mp.sqrt(5))/1024))
    print("b32={}".format(( 3785-1620*mp.sqrt(5))/1024))
    print("b41={}".format((-3365+2094*mp.sqrt(5))/6040))
    print("b42={}".format(( -975-3046*mp.sqrt(5))/2552))
    print("b43={}".format((467040+203968*mp.sqrt(5))/240845))
    print("g1 ={}".format((263+24*mp.sqrt(5))/1812))
    print("g2 ={}".format((125-1000*mp.sqrt(5))/3828))
    print("g3 ={}".format(1024*(3346+1623*mp.sqrt(5))/5924787))
    print("g4 ={}".format((30-4*mp.sqrt(5))/123))
def subarray(r=None,N=None,M=None):
    """Sub-array of an n-d array.

An n-d array is specified with a reference point and its shape.
The reference point is used to define an sub-array of an array as the
current array.
The reference point of a 1-d array is its first element;
the reference point of a 2-d array is its top-left element.

Syntax:
r_sub, n_sub, idx_q = subarray(r,N,M)

Inputs:
r is the reference point of the current array.
N is the shape of the current array.
M is the minimum dividable dimension. For example, if M is set to 3, an
array will be divisible into two halves along its dimensions greater than 3.

Returns:
r_sub is the reference points of all subarrays.
n_sub is the shape of subarrays.
idx_q is the central points of all subarrays.
"""
    N = np.double(N)
    ND = N.size
    if r is None:
        r = [0] * ND
    r = np.double(r)
    if r.size != ND:
        raise StandardError(u'User specified reference point of the array is invalid.')
    if M is None:
        M = 3
    m = np.double([0] * ND)
    n = np.double([0] * ND)
    idxQ = [[]]
    for i in range(0,ND):
        if N[i] <= M:
            n[i] = 1
            for j in range(0,len(idxQ)):
                for k in range(1,np.int(N[i])):
                    tmp = list(idxQ[j])
                    tmp.append(r[i]+k)
                    idxQ.append(tmp)
                idxQ[j].append(r[i])
        else:
            m[i] = np.ceil(N[i]/4.0-0.5)
            n[i] = m[i]*2.0 + 1.0
            for j in range(0,len(idxQ)):
                tmp = list(idxQ[j])
                tmp.append(r[i] + N[i] - m[i] - 1.0)
                idxQ[j].append(r[i] + m[i])
                idxQ.append(tmp)
    rsub = []
    idx_q = []
    for i in range(0,len(idxQ)):
        rsub.append(tuple(np.double(idxQ[i])-m))
        idx_q.append(tuple(idxQ[i]))
    return rsub,tuple(n),idx_q

def norm(v,axis=None):
    """Euclidean norm of 3D vector(s) v.

v can be input in the following ways:
1, a single vector v = np.array([x,y,z]);
2, an array of vectors, e.g.,:
  v = np.array([[x0,x1,...],
                [y0,y1,...],
                [z0,z1,...]])
  In this way elements of v along the first axis are summed up as
  orthogonal components.
A parameter ''axis'' can be provided to override the default convention.
"""
    if axis is None:
        axis = 0
    if np.shape(v)[axis] != 3:
        raise StandardError(u'Invalid vectors array or axis.')
    return np.sqrt(np.sum(np.power(v,2.0),axis=axis))

def direction(v,axis=None):
    """Direction of given vector(s).

u = direction(v,axis=a)
v is vector(s). Unit vector(s) u is direction of v.
Elements of v along axis a are its x-y-z components.
"""
    if axis is None:
        axis = 0
    if np.shape(v)[axis] != 3:
        raise StandardError(u'Invalid vectors array or axis.')
    return v / np.sqrt(np.sum(np.power(v, 2.0), axis=axis))

def as_xyz(a):
    """Convert given coordinate a into x-y-z form.

"""
    if np.shape(a)[0] == 3:
        return np.array(a)
    elif np.shape(a)[0] == 2:
        return ptr2xyz(a[0],a[1])
    else:
        raise StandardError(u'Input coordinate is invalid')

def xyz2ptr(x, y, z):
    """Convert Cartesian coordinate (x, y, z) to spherical coordinate (phi, theta, rho) with numexpr implemented fast evaluation.

(x, y, z) is Cartesian coordinate.
(phi, theta, rho) is returned spherical coordinate,
where phi is longitude in the range [-pi, pi] in radian,
theta is latitude in the range [-pi/2, pi/2] in radian,
and rho is the radial distance.
"""
    rho   = ne.evaluate("sqrt(x**2+y**2+z**2)")
    phi   = ne.evaluate("arctan2(y,x)")
    theta = ne.evaluate("arctan2(z,sqrt(x**2+y**2))")
    return phi, theta, rho

def ptr2xyz(phi, theta, rho=1.0):
    """Convert spherical coordinate (phi, theta, rho) to Cartesian coordinate (x, y, z) with numexpr implemented fast evaluation.

(phi, theta, rho) is the spherical coordinate,
where phi is longitude in the range [-pi, pi] in radian,
theta is latitude in the range [-pi/2, pi/2] in radian,
and rho is the radial distance.
(x, y, z) is returned Cartesian coordinate.
"""
    xy = ne.evaluate("rho*cos(theta)")
    x  = ne.evaluate("xy*cos(phi)")
    y  = ne.evaluate("xy*sin(phi)")
    z  = ne.evaluate("rho*sin(theta)")
    return x, y, z

def rotmat(axis, angle):
    """Rotation matrix.

v = R * u,
where u and v are 3-D vectors and R is the rotation matrix.
R rotates u arround an axis by an angle, yielding v.
"""
    if (np.sum(np.power(axis, 2.0)) <= DEPS) | (np.abs(angle) <= DEPS):
        return np.eye(3)
    axis = np.double(axis) / np.sqrt(np.sum(np.power(axis, 2.0)))
    a = np.cos(angle)
    b = np.sin(angle)
    c = 1 - a
    return np.matrix([
        [axis[0]*axis[0]*c + a, axis[0]*axis[1]*c - axis[2]*b, axis[0]*axis[2]*c + axis[1]*b],
        [axis[1]*axis[0]*c + axis[2]*b, axis[1]*axis[1]*c + a, axis[1]*axis[2]*c - axis[0]*b],
        [axis[2]*axis[0]*c - axis[1]*b, axis[2]*axis[1]*c + axis[0]*b, axis[2]*axis[2]*c + a]])

def triangle_area(v0,v1,v2):
    """Triangle area.

v0, v1 and v2 are vertices.
"""
    a2 = ((v0-v1)**2).sum(axis=0)
    b2 = ((v1-v2)**2).sum(axis=0)
    c2 = ((v2-v0)**2).sum(axis=0)
    ab = np.sqrt(a2*b2)
    if (np.abs(ab) <= DEPS).any():
        raise StandardError(u'Overlapping vertices error.')
    cos_v1 = (a2+b2-c2)/ab/2.0
    if np.isscalar(ab):
        if np.abs(cos_v1) >= 1:
            return 0.0
        else:
            sin_v1 = np.sqrt(1.0 - cos_v1**2)
            return 0.5*sin_v1*ab
    else:
        idx = (np.abs(cos_v1) < 1)
        sin_v1 = np.zeros(ab.shape)
        sin_v1[idx] = np.sqrt(1.0 - cos_v1[idx]**2)
        return 0.5*sin_v1*ab

def spherical_square_area(a):
    """Area of spherical square.

a is the span of the central arc of the square (in rad.).
"""
    tana = np.tan(a/2.0)
    return 4.0 * tana * np.arcsin(np.sin(a/2.0)*np.abs(tana)/np.sqrt(tana**2+1.0))/np.abs(tana)

def spherical_rectangle_area(a,b):
    """Area (solid angle) of spherical rectangle, in steradian.

A spherical rectangle consists two identical spherical triangles.
Area of spherical triangles = (sum of inner angles) - PI

a and b are the vertical and horizontal central arcs of the rectangle
(in rad.).

Reference:
https://en.wikipedia.org/wiki/Spherical_geometry
    """
    p = np.double([np.sin(a/2.0), 0.0,           np.cos(a/2.0)])
    q = np.double([np.sin(b/2.0), np.cos(b/2.0), 0.0          ])
    c = np.pi - np.arcsin(norm(np.cross(p,q)) / (norm(p) * norm(q)))
    return 4.0*c - 2.0*np.pi
