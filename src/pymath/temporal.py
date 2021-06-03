"""Temporal data analysis functions.

"""

## python 2 and 3 compatible
##

from pymath.common import *

def dst_b3(X,J=None,S=None):
    """1D discrete scale transform.

X is the 1D array.
J is the maximum scale.
S is the result.
"""
    h = np.double([1,4,6,4,1])/16.0
    if J is None:
        J = np.int64(np.ceil(np.log2(np.size(X)))) #transform to maximum scale by default
    if J <= 0:
        return X,np.zeros(np.shape(X))
    d = np.size(X)
    if S is None:
        S = np.zeros((J+1,d))
    S[0,:] = X[:]
    for k in range(J):
        S[k+1,:] += np.roll(S[k,:], (2**k)*2)*h[0]
        S[k+1,:] += np.roll(S[k,:], (2**k)  )*h[1]
        S[k+1,:] +=         S[k,:]           *h[2]
        S[k+1,:] += np.roll(S[k,:],-(2**k)  )*h[3]
        S[k+1,:] += np.roll(S[k,:],-(2**k)*2)*h[4]
    return S
