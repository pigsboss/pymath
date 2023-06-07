#!/usr/bin/env python
#coding=utf-8
"""Statistics related functions and helpers.

copyright: pigsboss@github
"""
import numpy as np

def spherical_random_sampling(nsamples, longitude_range=(0., 2.*np.pi), colatitude_range=(0., np.pi)):
    """Generate random samplings on unit sphere.
    Samples are uniformly distributed on the unit sphere per area.

    Arguments:
    nsamples is number of samples to return.
    longitude_range (phi_min, phi_max) is range of longitude, in rad.
    colatitude_range (theta_min, theta_max) is range of colatitude,
    i.e., the angle between polar axis, from north (+z) to south
    (-z), in [0, np.pi].

"""
    phi_min, phi_max     = longitude_range
    theta_min, theta_max = colatitude_range
    phi   = np.random.rand(nsamples) * (phi_max - phi_min) + phi_min
    theta = np.arccos(
        np.cos(theta_min) -
        np.random.rand(nsamples)*(np.cos(theta_min)-np.cos(theta_max)))
    return phi, theta

def float_to_integer(X, bits=16):
    """Float-point to integer scaling.

I = int((X - X_min) / (X_max - X_min) * (2**bits-1) + 0.5)
"""
    xmin = np.min(X)
    xmax = np.max(X)
    return np.int64((X-xmin)/(xmax-xmin)*(2**bits-1)+0.5)

def shannons(X):
    """Information entropy in bits of a sample X, given
its probability mass function P.

H(X) = - sum(P(X_i) * log2(P(X_i)), i)
"""
    cts = np.bincount(X)
    idx = np.nonzero(cts)
    pmf = cts[idx] / np.sum(cts)
    return -np.sum(pmf*np.log2(pmf))

def betadist(vmin, vmax, vmod, vstd, N):
    """Generate random values from a scale-and-shift beta distribution with given alpha.
vmin - minimum of values.
vmax - maximum of values.
vmod - mode (vmin < vmod < vmax) of values.
vstd - approximate standard deviation of values, in (0, sqrt(1/12)) for a,b > 1.
N    - number of random values.
"""
    a = (1./((vstd/(vmax-vmin))**2.)-4.)/8.
    m = (vmod-vmin)/(vmax-vmin)
    b = (a*(1-m)+2.*m-1)/m
    return np.random.beta(a,b,N)*(vmax-vmin)+vmin
