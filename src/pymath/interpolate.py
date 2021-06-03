"""Interpolation module with extra features.

"""

import os
import numpy as np
from pymath.linsolvers import tridiagonal
from os import path
from tempfile import mkdtemp

def bisection(xx, x):
    """Binary search to find the section where the coordinate falls.

Syntax:
l, u = bisection(xx, x)

xx is a sorted sequence as x[0], x[1], x[2], ..., x[n]
x  is an ND-array of coordinates

bisection tries to find the section (x[i], x[i+1]) so that x[i] <= x <= x[i+1]
for each x.

l and u are the indices of the lower end and upper end of the required section.
If x<x[0], u=-1. If x>x[n], l=n.
"""
    jl    = np.zeros(x.shape, dtype='int64') - 1
    ju    = np.zeros(x.shape, dtype='int64') + xx.size
    ascnd = np.bool8(xx[-1] >= xx[0])
    while np.any((ju-jl) > 1):
        jm  = (ju+jl)/2;
        idx = ((x >= xx[jm]) == ascnd)
        jl[idx]  = jm[idx]
        ju[~idx] = jm[~idx]
    return jl, ju

def is_regular(x):
    """Tell if a grid is regular or not.

"""
    dx = np.diff(x)
    return np.all(dx == dx[0])

class qslerp(object):
    """Sperical Linear Interpolation for Quaternions

"""
    def __init__(self,t,q,regular=None):
        self.t = t
        self.nknots = t.size
        if regular is None:
            self.is_regular = is_regular(self.t)
        else:
            self.is_regular = regular
        if self.is_regular:
            self.t0 = t[0]
            self.dt = t[1] - t[0]
        cos_omega  = q[0]
        sin_omega  = np.sqrt(np.sum(np.power(q[1:], 2.0), axis=0))
        omega      = np.arctan2(sin_omega, cos_omega)
        idx        = (sin_omega == 0.0)
        nu         = q[1:]
        nu[:,idx]  = np.reshape((1,0,0), (3,1))
        nu[:,~idx] = nu[:,~idx] / sin_omega[~idx]
        self.nu    = nu / np.sqrt(np.sum(np.power(nu, 2.0), axis=0))
        self.omega = omega
        self.nuome = self.nu * self.omega

    def __call__(self,t):
        t     = np.array(t, ndmin=1)
        omega = np.empty(t.shape, dtype=self.omega.dtype)
        nu    = np.empty((3,)+t.shape, dtype=self.nu.dtype)
        nuome = np.empty((3,)+t.shape, dtype=self.nu.dtype)
        q     = np.empty((4,)+t.shape, dtype=self.nu.dtype)
        if self.is_regular:
            jj = (t-self.t0) / self.dt
            jl = np.array(np.floor(jj), dtype='int64', ndmin=1)
            ju = jl + 1
            inside = np.logical_and(jl>=0, ju<self.nknots)
            A = ju[inside] - jj[inside]
            B = jj[inside] - jl[inside]
        else:
            jl,ju = bisection(self.x, x)
            inside = np.logical_and(jl>=0, ju<self.nknots)
            A = (self.t[ju[inside]]-t[inside]) / self.dt[jl[inside]]
            B = (t[inside]-self.t[jl[inside]]) / self.dt[jl[inside]]
        nuome[:,inside] = A*self.nuome[:,jl[inside]] + B*self.nuome[:,ju[inside]]
        omega[inside] = np.sqrt(np.sum(np.power(nuome[:,inside], 2.0), axis=0))
        iszero = (omega == 0.0)
        nu[:,np.logical_and(inside,iszero)] = np.reshape((1,0,0), (3,1))
        nu[:,np.logical_and(inside,np.logical_not(iszero))] = nuome[:,np.logical_and(inside,np.logical_not(iszero))]/omega[np.logical_and(inside,np.logical_not(iszero))]
        q[0,inside] = np.cos(omega[inside])
        q[1:,inside] = np.sin(omega[inside])*nu[:,inside]
        q[:,~inside] = np.nan
        return q


class qscserp(qslerp):
    def __init__(self,t,q,regular=None):
        qslerp.__init__(self,t,q,regular=regular)
        n = self.nknots - 1
        h = np.diff(t)
        a = np.copy(h[:-1])
        c = np.copy(h[1:])
        b = 2.0*(a+c)
        d = 6.0*((self.nuome[:,2:]-self.nuome[:,1:-1])/h[1:] - (self.nuome[:,1:-1]-self.nuome[:,:-2])/h[:-1])
        m = np.zeros((3, n+1))
        m[:,1:-1] = tridiagonal(a, b, c, d)
        self.B = (self.nuome[:,1:]-self.nuome[:,:-1])/h - h*m[:,:-1]/2.0 - h*(m[:,1:]-m[:,:-1])/6.0
        self.C = 0.5*m[:,:-1]
        self.D = (m[:,1:] - m[:,:-1])/h/6.0

    def __call__(self,t):
        t     = np.array(t, ndmin=1)
        omega = np.empty(t.shape, dtype=self.omega.dtype)
        nu    = np.empty((3,)+t.shape, dtype=self.nu.dtype)
        nuome = np.empty((3,)+t.shape, dtype=self.nu.dtype)
        q     = np.empty((4,)+t.shape, dtype=self.nu.dtype)
        if self.is_regular:
            jj = (t-self.t0) / self.dt
            jl = np.array(np.floor(jj), dtype='int64', ndmin=1)
            ju = jl + 1
            inside = np.logical_and(jl>=0, ju<self.nknots)
        else:
            jl,ju = bisection(self.x, x)
            inside = np.logical_and(jl>=0, ju<self.nknots)
        tt = t[inside] - self.t[jl[inside]]
        nuome[:,inside] = self.nuome[:,jl[inside]] + self.B[:,jl[inside]]*tt + self.C[:,jl[inside]]*(tt**2.0) + self.D[:,jl[inside]]*(tt**3.0)
        omega[inside] = np.sqrt(np.sum(np.power(nuome[:,inside], 2.0), axis=0))
        iszero = (omega == 0.0)
        nu[:,np.logical_and(inside, iszero)] = np.reshape((1,0,0), (3,1))
        nu[:,np.logical_and(inside,~iszero)] = nuome[:,np.logical_and(inside,~iszero)]/omega[np.logical_and(inside,~iszero)]
        q[0,inside]  = np.cos(omega[inside])
        q[1:,inside] = np.sin(omega[inside])*nu[:,inside]
        q[:,~inside] = np.nan
        return q
        

class interp1d(object):
    def __init__(self,x,y,kind='linear',container=None,buffer_size=32*1024**2,temporary=True,regular=True):
        if container is None:
            self.container = mkdtemp()
            self.is_temporary = True
        else:
            self.is_temporary = temporary
            if temporary:
                self.container = mkdtemp(dir=container)
            else:
                os.mkdir(container)
                self.container = container
        self.kind = kind
        self.nknots = y.size
        self.nintvs = y.size-1
        self.y_file = path.join(self.container, 'y.dat')
        self.y  = np.memmap(self.y_file, shape=(self.nknots,),dtype=y.dtype,mode='w+')
        self.is_regular = regular
        nbuf = buffer_size / np.dtype(x.dtype).itemsize
        t = 0
        if self.is_regular:
            self.x_0 = x[0]
            self.dx = x[1] - x[0]
            while t < self.nintvs:
                n = min(nbuf, int(self.nintvs-t))
                self.y [t:t+n] = y[t:t+n]
                t += n
        else:
            self.x_file = path.join(self.container, 'x.dat')
            self.dx_file = path.join(self.container, 'dx.dat')
            self.x  = np.memmap(self.x_file, shape=(self.nknots,),dtype=x.dtype,mode='w+')
            self.dx = np.memmap(self.dx_file,shape=(self.nintvs,),dtype=x.dtype,mode='w+')
            while t < self.nintvs:
                n = min(nbuf, int(self.nintvs-t))
                self.x [t:t+n] = x[t:t+n]
                self.y [t:t+n] = y[t:t+n]
                self.dx[t:t+n] = x[t+1:t+n+1] - x[t:t+n]
                t += n
            self.x[-1] = x[-1]
        self.y[-1] = y[-1]

    def __del__(self):
        if self.is_regular:
            del self.x_0
        else:
            del self.x
        del self.y
        del self.dx
        if self.is_temporary:
            os.remove(self.y_file)
            if not self.is_regular:
                os.remove(self.x_file)
                os.remove(self.dx_file)
            os.removedirs(self.container)

    def __call__(self,x):
        x = np.array(x,ndmin=1)
        y = np.empty(x.shape,dtype=self.y.dtype)
        if self.kind == 'linear':
            if self.is_regular:
                jj = (x-self.x_0) / self.dx
                jl = np.array(np.floor(jj),dtype='int64',ndmin=1)
                ju = jl+1
                A = ju-jj
                B = jj-jl
                rng = np.logical_and(jl>=0, ju<self.nknots)
                y[rng] = A[rng]*self.y[jl[rng]] + B[rng]*self.y[ju[rng]]
                y[~rng] = np.nan
            else:
                jl,ju = bisection(self.x, x)
                rng = np.logical_and(jl>=0, ju<self.nknots)
                A = (self.x[ju[rng]]-x[rng]) / self.dx[jl[rng]]
                B = (x[rng]-self.x[jl[rng]]) / self.dx[jl[rng]]
                y[rng] = A*self.y[jl[rng]] + B*self.y[ju[rng]]
                y[~rng] = np.nan
        elif self.kind == 'nearest':
            if self.is_regular:
                jj = np.uint64(np.clip(np.round((x-self.x_0) / self.dx), 0, self.nknots-1))
                y[:] = self.y[jj]
            else:
                jl,ju = bisection(self.x, x)
                mid = np.logical_and(jl>=0, ju<self.nknots)
                pos = np.bool_(jl<0)
                neg = np.bool_(ju>=self.nknots)
                y[neg] = self.y[0]
                y[pos] = self.y[-1]
                ascnd = np.bool_(self.x[-1] >= self.x[0])
                xl = self.x[jl[mid]]
                xu = self.x[ju[mid]]
                lef = np.bool_(((xu-x[mid]) >= (x[mid]-xl)) == ascnd)
                y[mid[lef]] = self.y[jl[lef]]
                y[mid[~lef]] = self.y[ju[~lef]]
        return y
