"""Linear Equations System Solvers
"""

## python 2 and 3 compatible
##

import numpy as np
import warnings

def lines_intersection(list_of_r, list_of_u, in_dtype='float128'):
    """Find line-line intersections or nearest points to skew lines.
    line 1: r = r1 + u1*t
    line 2: r = r2 + u2*t
    ...
    
    Each r is a (N, 3) array.
    Each u is a (N, 3) array and each row of it represents a unit vector.

    Returns:
    p - (N, 3) array, each row represents an intersection.
        np.nan returned if the two lines are parallel.
    s - d1**2 + d2**2 + ..., where d[i] is distance between p and the
        i-th line.
    """
    N = len(list_of_r)
    M = list_of_r[0].shape[0]
    A = np.zeros((M,3,3), dtype=in_dtype)
    b = np.zeros((M,3,1), dtype=in_dtype)
    for i in range(N):
        r_in = list_of_r[i].astype(in_dtype)
        u_in = list_of_u[i].astype(in_dtype)
        U = np.eye(3, dtype=in_dtype) - \
            np.matmul(
                np.reshape(u_in, (-1,3,1)),
                np.reshape(u_in, (-1,1,3)),
                axes=[(-2,-1), (-2,-1), (-2,-1)])
        A[:] += U
        b[:] += np.matmul(
            U,
            np.reshape(r_in, (-1,3,1)),
            axes=[(-2,-1), (-2,-1), (-2,-1)])
    adjA = np.empty_like(A)
    adjA[:,0,0] = A[:,1,1]*A[:,2,2]-A[:,1,2]*A[:,2,1]
    adjA[:,0,1] = A[:,0,2]*A[:,2,1]-A[:,0,1]*A[:,2,2]
    adjA[:,0,2] = A[:,0,1]*A[:,1,2]-A[:,0,2]*A[:,1,1]
    adjA[:,1,0] = A[:,1,2]*A[:,2,0]-A[:,1,0]*A[:,2,2]
    adjA[:,1,1] = A[:,0,0]*A[:,2,2]-A[:,0,2]*A[:,2,0]
    adjA[:,1,2] = A[:,0,2]*A[:,1,0]-A[:,0,0]*A[:,1,2]
    adjA[:,2,0] = A[:,1,0]*A[:,2,1]-A[:,1,1]*A[:,2,0]
    adjA[:,2,1] = A[:,0,1]*A[:,2,0]-A[:,0,0]*A[:,2,1]
    adjA[:,2,2] = A[:,0,0]*A[:,1,1]-A[:,0,1]*A[:,1,0]
    D = A[:,0,0]*adjA[:,0,0] + \
        A[:,0,1]*adjA[:,1,0] + \
        A[:,0,2]*adjA[:,2,0]
    p = np.empty((D.size, 3), dtype=in_dtype)
    D_is_0 = np.isclose(D, 0., atol=np.finfo(in_dtype).resolution**.5)
    if np.all(D_is_0):
        warnings.warn(
            "All determinants (max: {:.2E}) are close to zero.".format(
                np.max(np.abs(D))), RuntimeWarning)
    invA = adjA[~D_is_0]/np.reshape(D[~D_is_0], (-1,1,1))
    p[ D_is_0, :] = np.nan
    p[~D_is_0, :] = np.matmul(invA, b[~D_is_0]).reshape((-1,3))
    s = np.zeros((M,), dtype=in_dtype)
    for i in range(N):
        rmp = list_of_r[i].astype(in_dtype)-p
        s += np.sum(rmp**2., axis=-1) - np.sum(rmp*list_of_u[i].astype(in_dtype), axis=-1)**2.
    out_dtype = list_of_r[0].dtype
    return p.astype(out_dtype), s.astype(out_dtype)

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
