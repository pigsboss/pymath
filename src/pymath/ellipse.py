#!/usr/bin/env python
"""Elliptic orbit related functions.

nu      -  local true anomaly, in rad.
nu0     -  reference true anomaly, in rad.
dnu/dt  -  angular velocity, in rad/yr.

                      2 * pi * (1 + e*cos(nu))^2
           dnu/dt = -----------------------------
                      a^(3/2) * (1 - e^2)^(3/2)

omega   -  auxiliary term.
                     2*pi        (1 + e*cos(nu))^2
           dnu/dt = -------  x  -------------------
                    a^(3/2)       (1 - e^2)^(3/2)
             2*pi
           --------- depends on semi-major axis a.
            a^(3/2)

                   (1+e*cos(nu))^2
           omega = --------------- is orbit-size independent.
                    (1-e^2)^(3/2)

phi     -  auxiliary term.

                     / nu
                     |
                     |      1
           phi(nu) = | ----------- dnu'
                     |  omega(nu')
                     |
                     / 0

recome  -  omega reciprocal, i.e., 1 / omega. 

iphi    -  inverse phi, i.e., nu = iphi(phi(nu)).
"""

## python 2 and 3 compatible
##

import numpy as np
import pickle
from .quadrature import *
from scipy.interpolate import griddata, interp2d, RectBivariateSpline

def recome(e, nu):
    """Angular velocity auxiliary term reciprocal.

              (1 - e^2)^(3/2)
1/omega = ------------------------
             (1 + e*cos(nu))^2
"""
    return (1.0-e**2.0)**1.5 / (1.0 + e*np.cos(nu))**2.0

def phi_glq(e, nu):
    """Definite integral of angular velocity auxiliary reciprocal, numerical with GL quadrature.

e   - eccentricity of the elliptic orbit.
nu  - local true anomaly.
"""
    if nu <= np.pi:
        return nu*0.5*np.sum(recome(e, nu*0.5*GLQA64 + nu*0.5) * GLQW64)
    else:
        return np.pi + (nu-np.pi)*0.5*np.sum(recome(e, (nu-np.pi)*0.5*GLQA64 + (nu+np.pi)*0.5) * GLQW64)

def phi(e, nu):
    """Definite integral of angular velocity auxiliary reciprocal, analytical.

e   - eccentricity of the elliptic orbit.
nu  - local true anomaly.
"""
    return np.mod(np.real(
        -2j*np.arctanh(1j*np.tan(0.5*nu)*np.sqrt(1-e)/np.sqrt(1+e))-
        np.sin(nu)*e*np.sqrt(1-e**2)/(1+e*np.cos(nu))),2.0*np.pi)

def orbitab_glq(npts=64):
    """Orbit dynamics data table for interpolation, numerically integrated with quadrature.

Inputs:

npts  - number of interpolation points.


Returns:
ecc      - eccentricity
nu/t     - true anomaly, or time proportion
phivals  - phi(nu) values corresponding to nu
iphivals - iphi(t) values corresponding to t
"""
    egv        = np.arange(npts)*1.0/npts
    nugvu      = (np.arange(npts)    )*1.0/npts * np.pi
    nugvd      = (np.arange(npts)+1.0)*1.0/npts * np.pi + np.pi
    e,  nuu, w = np.meshgrid(egv, nugvu, GLQW64)
    e,  nud, _ = np.meshgrid(egv, nugvd, GLQW64)
    _,  _,   a = np.meshgrid(egv, nugvu, GLQA64)
    phivalsu   = np.clip(np.sum(w * nuu*0.5 * recome(e, nuu*0.5*a + nuu*0.5), axis=2), 0.0, np.pi)
    phivalsd   = np.pi + np.clip(np.sum(w * (nud-np.pi)*0.5 * recome(e, (nud-np.pi)*0.5*a + (nud+np.pi)*0.5), axis=2), 0.0, np.pi)
    phivals    = np.concatenate((phivalsu, phivalsd), axis=0)
    nu         = np.concatenate((nuu, nud), axis=0)
    e          = np.concatenate((e, e), axis=0)
    iphivals   = griddata((e[:,:,0].ravel(), phivals.ravel()),
                          nu[:,:,0].ravel(),
                          (e[:,:,0].ravel(), nu[:,:,0].ravel()),
                          method = 'cubic',
                          fill_value = np.pi*2.0)
    return {'ecc':e[:,:,0], 'nu':nu[:,:,0], 'phi':phivals, 'iphi':iphivals.reshape(phivals.shape)}

def orbitab(npts=64):
    """Orbit dynamics data table for interpolation, semi-analytically integrated.

Inputs:

npts  - number of interpolation points.


Returns:
ecc      - eccentricity
nu/t     - true anomaly, or time proportion
phivals  - phi(nu) values corresponding to nu
iphivals - iphi(t) values corresponding to t
"""
    egv      = np.arange(npts)*1.0/npts
    nugv     = np.arange(2*npts)*1.0/(2*npts-1) * 2.0 * np.pi
    e, nu    = np.meshgrid(egv, nugv)
    phivals  = phi(e, nu)
    iphivals = griddata((e.ravel(), phivals.ravel()),
                          nu.ravel(),
                          (e.ravel(), nu.ravel()),
                          method = 'cubic',
                          fill_value = np.pi*2.0)
    return {'ecc':e, 'nu':nu, 'phi':phivals, 'iphi':iphivals.reshape(phivals.shape)}
