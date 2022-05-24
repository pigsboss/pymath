"""Linear Equations System Solvers
"""

## python 2 and 3 compatible
##

import numpy as np

def two_lines_intersection(r1, u1, r2, u2):
    """Find line-line intersections or nearest points to skew lines.
    line 1: r = r1 + u1*t
    line 2: r = r2 + u2*t

    r1 and r2 are (N, 3) arrays.
    u1 and u2 are (N, 3) arrays and each row represents a unit vector.

    Return r as (N, 3) array, each row represents an intersection.
    np.nan returned if the two lines are parallel.
    """
    U1 = np.eye(3) - np.matmul(
        np.reshape(u1, (-1,3,1)),
        np.reshape(u1, (-1,1,3)),
        axes=[(-2,-1), (-2,-1), (-2,-1)])
    U2 = np.eye(3) - np.matmul(
        np.reshape(u2, (-1,3,1)),
        np.reshape(u2, (-1,1,3)),
        axes=[(-2,-1), (-2,-1), (-2,-1)])
    A = U1+U2
    b = np.matmul(U1, np.reshape(r1,(-1,3,1)), axes=[(-2,-1),(-2,-1),(-2,-1)]) + \
        np.matmul(U2, np.reshape(r2,(-1,3,1)), axes=[(-2,-1),(-2,-1),(-2,-1)])
    D = np.linalg.det(A)
    r = np.empty((D.size, 3))
    D_is_0 = np.isclose(D, 0.)
    r[ D_is_0, :] = np.nan
    r[~D_is_0, :] = np.matmul(np.linalg.inv(A[~D_is_0]), b[~D_is_0]).reshape((-1,3))
    return r

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
