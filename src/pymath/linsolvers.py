"""Linear Equations System Solvers
"""

## python 2 and 3 compatible
##

import numpy as np

def least_squares(A, b):
    """Solving overdetermined system Ax=b using least-squares.
A is N x M coefficient matrix (N observations, M variables).
x is M x 1 vector (M unknown parameters).
b is N x 1 vector (N observed outcomes).

Return x, x_error
"""
    n, m = np.shape(A)
    P = (np.matrix(A.T) * np.matrix(A))**-1
    x = P * np.matrix(A.T) * np.matrix(b.reshape((-1,1)))
    S = np.sum((b - np.array(np.matrix(A)*x).ravel())**2.0)
    err = np.sqrt(S/(n-m)*np.diag(P))
    return np.array(x).ravel(), err.ravel()

def weighted_least_squares(A, b, W=None):
    """Solving overdetermined system Ax=b using least-squares in case the observations are weighted.
A is N x M coefficient matrix (N observations, M variables).
x is M x 1 vector (M unknown parameters).
b is N x 1 vector (N observed outcomes).
W is N x N diagonal matrix (N weights for each of the N observations).

Return x, x_error
"""
    if W is None:
        return least_squares(A, b)
    n, m = np.shape(A)
    if np.ndim(W) == 1:
        W = np.diag(W)
    P = np.matrix(A.T)*np.matrix(W)
    Q = (P*np.matrix(A))**-1
    x = Q*P*np.matrix(b.reshape(-1,1))
    err = np.sqrt(np.diag(Q))
    return np.array(x).ravel(), err.ravel()

def tridiagonal(a, b, c, d, x=None):
    """Tri-diagonal system solver.

|   1                     |   |   0 |   |   0 |
| a_0  b_0  c_0           |   | x_0 |   | d_0 |
|      a_1  b_1  c_1      | X | x_1 | = | d_1 |
|        .    .    .      |   |   . |   |   . |
|             .    .    . |   |   . |   |   . |

Boundary condition: natural (2nd order derivative is zero).
"""
    n  = np.size(a)
    if np.ndim(d) == 1:
        d = np.reshape(d, (1, n))
    if x is None:
        x = np.empty(d.shape, dtype=d.dtype)
    c[0]   = c[0]/b[0]
    d[:,0] = d[:,0]/b[0]
    for i in range(1, n):
        tmp = b[i] - a[i]*c[i-1]
        c[i]   = c[i]/tmp
        d[:,i] = (d[:,i] - a[i]*d[:,i-1])/tmp
    x[:,-1] = d[:,-1]
    for i in range(n-2, -1, -1):
        x[:,i] = d[:,i] - c[i]*x[:,i+1]
    return x
