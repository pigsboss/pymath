"""Numerical function helpers.

"""

## python 2 and 3 compatible
##

from pymath.common import *
import pymath.sphere as sp
from scipy.integrate import quad,dblquad,tplquad
import scipy.interpolate as interpolate
import scipy.misc

def multiply(*args):
    '''Combine functions by multiplying them one after another.

Syntax:
add((func1,nargs1),(func2,nargs2),(func3,nargs3),...)
'''
    return reduce(multiply2,args)

def multiply2(a,b):
    if a[1]>0:
        if b[1]>0:
            return lambda *args:a[0](*tuple(args[0:a[1]]))*\
                b[0](*tuple(args[a[1]:a[1]+b[1]])), a[1]+b[1]
        else:
            return lambda *args:a[0](*tuple(args[0:a[1]]))*b[0], a[1]
    else:
        if b[1]>0:
            return lambda *args:a[0]*b[0](*tuple(args[0:b[1]])), b[1]
        else:
            return a[0]*b[0], a[1]+b[1]

def add(*args):
    '''Combine functions by adding them one after another.

Syntax:
add((func1,nargs1),(func2,nargs2),(func3,nargs3),...)
'''
    return reduce(add2,args)

def add2(a,b):
    if a[1]>0:
        if b[1]>0:
            return lambda *args:a[0](*tuple(args[0:a[1]]))+\
                b[0](*tuple(args[a[1]:a[1]+b[1]])), a[1]+b[1]
        else:
            return lambda *args:a[0](*tuple(args[0:a[1]]))+b[0], a[1]
    else:
        if b[1]>0:
            return lambda *args:a[0]+b[0](*tuple(args[0:b[1]])), b[1]
        else:
            return a[0]+b[0], a[1]+b[1]

def integral2d(func,ylimits,xlimits):
    if callable(ylimits[0]):
        gfun = ylimits[0]
    else:
        ymin = ylimits[0]
        gfun = lambda x:ymin
    if callable(ylimits[1]):
        hfun = ylimits[1]
    else:
        ymax = ylimits[1]
        hfun = lambda x:ymax
    return dblquad(func,xlimits[0],xlimits[1],gfun,hfun)

def integral3d(func,zlimits,ylimits,xlimits):
    if callable(ylimits[0]):
        gfun = ylimits[0]
    else:
        ymin = ylimits[0]
        gfun = lambda x:ymin
    if callable(ylimits[1]):
        hfun = ylimits[1]
    else:
        ymax = ylimits[1]
        hfun = lambda x:ymax
    if callable(zlimits[0]):
        qfun = zlimits[0]
    else:
        zmin = zlimits[0]
        qfun = lambda x,y:zmin
    if callable(zlimits[1]):
        rfun = zlimits[1]
    else:
        zmax = zlimits[1]
        rfun = lambda x,y:zmax
    return tplquad(func,xlimits[0],xlimits[1],gfun,hfun,qfun,rfun)

def measure(limits,space='euclidean'):
    if space == 'euclidean':
        if np.size(limits) > 2:
            m = reduce(lambda a,b:np.abs(a[1]-a[0])*np.abs(b[1]-b[0]),limits)
        else:
            limits = np.ravel(limits)
            m = np.abs(limits[1] - limits[0])
    elif space == 'spheric':
        if len(limits) == 2:
            try:
                lon = limits['longitude']
                lat = limits['latitude']
            except KeyError:
                lon = limits[0]
                lat = limits[1]
            m = (np.sin(lat[1]) - np.sin(lat[0])) * (lon[1] - lon[0])
        elif len(limits) == 3:
            try:
                lon = limits['longitude']
                lat = limits['latitude']
                rad = limits['radius']
            except KeyError:
                lon = limits[0]
                lat = limits[1]
                rad = limits[2]
            m = (np.sin(lat[1]) - np.sin(lat[0])) * (lon[1] - lon[0]) * \
                (np.power(rad[1],3.0) - np.power(rad[0],3.0)) / 3.0
    elif space == 'polar':
        try:
            phi = limits['angle']
            rad = limits['radius']
        except KeyError:
            phi = limits[0]
            rad = limits[1]
        m = (np.power(rad[1],2.0) - np.power(rad[0],2.0)) / 2.0 * \
            (phi[1] - phi[0])
    else:
        raise StandardError('Unsupported space' + str(space))
    return m

class function(object):
    '''Function defined in 1-D Euclidean space.

'''
    def __init__(self,func):
        '''func is either a callable object or a constant.
        '''
        if callable(func):
            self.__function__ = func
        else:
            self.__function__ = lambda *args:func
    def __call__(self,*args):
        return self.__function__(*args)
    def derivative(self,x,dx=1.0):
        return scipy.misc.derivative(self.__function__,x,dx=dx)
    def integral(self,*args):
        """Numerical integral.

"""
        limits = np.ravel(args)
        return quad(self.__function__,limits[0],limits[1])[0]

class function2d(function):
    '''Function defined in 2-D Euclidean space.

'''
    def derivative(self,x,y,dx=1.0,dy=1.0):
        fx = lambda x:self.__function__(x,y)
        fy = lambda y:self.__function__(x,y)
        return scipy.misc.derivative(fx,x,dx=dx),\
            scipy.misc.derivative(fy,y,dx=dy)
    def integral(self,*args):
        '''Numerical integral.

Syntax:
integral(ylimits, xlimits),
where ylimits are literal limits (e.g., [lower_lim, upper_lim]),
xlimits are either literal or functions of y.
'''
        return integral2d(self.__function__,*args)[0]

class function3d(function):
    '''Function defined in 3-D Euclidean space.

'''
    def derivative(self,x,y,z,dx=1.0,dy=1.0,dz=1.0):
        fx = lambda x:self.__function__(x,y,z)
        fy = lambda y:self.__function__(x,y,z)
        fz = lambda z:self.__function__(x,y,z)
        return scipy.misc.derivative(fx,x,dx=dx),\
            scipy.misc.derivative(fy,y,dx=dy),\
            scipy.misc.derivative(fz,z,dx=dz)
    def integral(self,*args):
        '''Numerical integral.

Syntax:
integral(zlimits, ylimits, xlimits),
where zlimits are literal limits (e.g., [lower_lim, upper_lim]),
ylimits are either literal or functions of z,
xlimits are either literal of runctions of z and y.
'''
        return integral3d(self.__function__,*args)[0]

class constant(function):
    def __init__(self,constant_value=None,space='euclidean'):
        self.constant_value = constant_value
        self.space = space
    def __call__(self,*args):
        return self.constant_value
    def derivative(self,*args):
        return 0
    def integral(self,*args):
        return measure(*args,space=self.space)

class delta(function):
    def __init__(self):
        pass
    def __call__(self,arg):
        if np.abs(arg)<=DEPS:
            return np.inf
        else:
            return 0
    def derivative(self,arg):
        if np.abs(arg)<=DEPS:
            return np.nan
        else:
            return 0
    def integral(self,*args):
        limits = np.ravel(args)
        if (limits[0]<0) & (limits[1]>0):
            return 1
        else:
            return 0

class gauss(function):
    def __init__(self,sigma):
        self.sigma = sigma
        self.__function__ = \
            lambda x:np.exp(-0.5*x**2.0/sigma**2.0)/\
            (np.sqrt(2.0*np.pi)*sigma)

class gauss2d(function2d):
    def __init__(self,sigma):
        self.sigma = sigma
        self.__function__ = \
            lambda x,y:np.exp(-0.5*(x**2.0+y**2.0)/sigma**2.0)/\
            (2.0*np.pi*sigma**2.0)

class interpolated(function):
    def __init__(self,x,f,**kwargs):
        self.x = x
        self.f = f
        self.__function__ = interpolate.interp1d(x,f,**kwargs)

class interpolated2d(function2d):
    def __init__(self,x,y,f,**kwargs):
        self.x = x
        self.y = y
        self.f = f
        self.__function__ = interpolate.interp2d(x,y,f,**kwargs)

class sphere2d(function):
    '''Function defined on 2-D unit sphere, called with longitude and latitude as its arguments.

'''
    def integral(self,*args):
        '''Numerical integral.

Syntax:
integral(lat_limits, lon_limits),
where lat_limits are literal limits (e.g., [lower_lim, upper_lim]),
lon_limits are either literal or functions of latitude.
'''
        func = lambda lon,lat:np.cos(lat)*self.__function__(lon,lat)
        return integral2d(func,*args)[0]

class sphere_gauss(sphere2d):
    def __init__(self,lon,lat,sigma):
        self.longitude = lon
        self.latitude = lat
        self.sigma = sigma
        self.__function__ = \
            lambda a,b:np.exp(-0.5*sp.distance([a,b],[lon,lat])**2.0/sigma**2.0)/\
            (2.0*np.pi*sigma**2)
