"""This module implements some quaternion related functions provided by MATLAB.

MATLAB conventions:
1. A quaternion is a 1-by-4 array.
2. A sequence of quaternions is an N-by-4 2D array.
"""

## python 2 and 3 compatible
##

from pymath.common import *
def quat2dcm(q):
    '''Convert quaternion to direction cosine matrix.

'''
    qin = quatnormalize(q)
    dcm = np.zeros((3,3,np.shape(qin)[0]),dtype='float64')

    dcm[0,0,:] = qin[:,0]**2.0 + qin[:,1]**2.0 - qin[:,2]**2.0 - qin[:,3]**2.0
    dcm[0,1,:] = 2.0*(qin[:,1]*qin[:,2] + qin[:,0]*qin[:,3])
    dcm[0,2,:] = 2.0*(qin[:,1]*qin[:,3] - qin[:,0]*qin[:,2])

    dcm[1,0,:] = 2.0*(qin[:,1]*qin[:,2] - qin[:,0]*qin[:,3])
    dcm[1,1,:] = qin[:,0]**2.0 - qin[:,1]**2.0 + qin[:,2]**2.0 - qin[:,3]**2.0
    dcm[1,2,:] = 2.0*(qin[:,2]*qin[:,3] + qin[:,0]*qin[:,1])

    dcm[2,0,:] = 2.0*(qin[:,1]*qin[:,3] + qin[:,0]*qin[:,2])
    dcm[2,1,:] = 2.0*(qin[:,2]*qin[:,3] - qin[:,0]*qin[:,1])
    dcm[2,2,:] = qin[:,0]**2.0 - qin[:,1]**2.0 - qin[:,2]**2.0 + qin[:,3]**2.0
    return dcm
def dcm2quat(dcm):
    '''Convert direction cosine matrix to quaternion.

'''
    d = np.diagonal(dcm)
    tr = np.sum(d,axis=1)
    M = np.shape(dcm)[2]
    q = np.zeros((M,4),dtype='float64')

    c0 = np.array(tr>0,dtype='bool')
    c1 = np.logical_and(~c0, np.logical_and(d[:,1]>d[:,0], d[:,1]>d[:,2]))
    c2 = np.logical_and(~c0, np.logical_and(~c1, d[:,2]>d[:,0]))
    c3 = np.logical_and(~c0, np.logical_and(~c1, ~c2))

    sq0 = np.zeros(M,dtype='float64')
    sq1 = np.zeros(M,dtype='float64')
    sq2 = np.zeros(M,dtype='float64')
    sq3 = np.zeros(M,dtype='float64')

    sq0[c0] = np.sqrt(tr[c0]+1.0)
    q[c0,0] = 0.5*sq0[c0]

    sq1[c1] = np.sqrt(d[c1,1]-d[c1,0]-d[c1,2]+1.0)
    q[c1,2] = 0.5*sq1[c1]
    idx = np.array(np.abs(sq1)>=DEPS, dtype='bool')
    sq1[idx] = 0.5 / sq1[idx]

    sq2[c2] = np.sqrt(d[c2,2]-d[c2,0]-d[c2,1]+1.0)
    q[c2,3] = 0.5*sq2[c2]
    idx = np.array(np.abs(sq2)>=DEPS, dtype='bool')
    sq2[idx] = 0.5 / sq2[idx]

    sq3[c3] = np.sqrt(d[c3,0]-d[c3,1]-d[c3,2]+1.0)
    q[c3,1] = 0.5*sq3[c3]
    idx = np.array(np.abs(sq3)>=DEPS, dtype='bool')
    sq3[idx] = 0.5 / sq3[idx]

    q[c0,1] = (dcm[1,2,c0]-dcm[2,1,c0]) / (2.0*sq0[c0])
    q[c0,2] = (dcm[2,0,c0]-dcm[0,2,c0]) / (2.0*sq0[c0])
    q[c0,3] = (dcm[0,1,c0]-dcm[1,0,c0]) / (2.0*sq0[c0])

    q[c1,0] = (dcm[2,0,c1]-dcm[0,2,c1]) * sq1[c1]
    q[c1,1] = (dcm[0,1,c1]+dcm[1,0,c1]) * sq1[c1]
    q[c1,3] = (dcm[1,2,c1]+dcm[2,1,c1]) * sq1[c1]

    q[c2,0] = (dcm[0,1,c2]-dcm[1,0,c2]) * sq2[c2]
    q[c2,1] = (dcm[2,0,c2]+dcm[0,2,c2]) * sq2[c2]
    q[c2,2] = (dcm[1,2,c2]+dcm[2,1,c2]) * sq2[c2]

    q[c3,0] = (dcm[1,2,c3]-dcm[2,1,c3]) * sq3[c3]
    q[c3,2] = (dcm[0,1,c3]+dcm[1,0,c3]) * sq3[c3]
    q[c3,3] = (dcm[2,0,c3]+dcm[0,2,c3]) * sq3[c3]
    return q
def quatnormalize(q):
    '''Normalize a quaternion.

n = quatnomalize(q) calculates the normalized quaternion, n, for a given
quaternion q. Input q is an M-by-4 matrix containign M quaternions.
n returns an M-by-4 matrix of normalized quaternions.
'''
    return np.array(q,ndmin=2) / np.array(quatmod(q),ndmin=2).T
def quatmod(q):
    '''Calculate the modules of a quaternion.

'''
    return np.sqrt(np.sum(np.array(q,ndmin=2)**2.0,axis=1))
