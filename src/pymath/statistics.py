#!/usr/bin/env python
#coding=utf-8
"""Statistics related functions and helpers.

copyright: pigsboss@github
"""
import numpy as np

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

def betadist(vmin, vmax, vmod, a, N):
    """Generate random values from a scale-and-shift beta distribution with given alpha.
vmin - minimum of values.
vmax - maximum of values.
vmod - mode (vmin < vmod < vmax) of values.
a    - alpha parameter of beta distribution.
N    - number of random values.
"""
    m = (vmod-vmin)/(vmax-vmin)
    b = (a*(1-m)+2.*m-1)/m
    return np.random.beta(a,b,N)*(vmax-vmin)+vmin
