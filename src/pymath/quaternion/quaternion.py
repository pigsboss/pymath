'''Quaternion and Euler's angles convention

A quaternion or a sequence of Euler angles (phi, theta, psi) rotates a rigid body from
its reference orientation to an arbitrary orientation, or, convert a 3D coordinate in
the rotated coordinate system to the original coordinate system.

For example, taking J2000 coordinate system as reference, i.e., the intertial frame of
reference, let the laboratory frame of reference align with J2000 frame.
A quaternion or a sequence of Euler angles are used to compute the coordinate of any point
of the lab frame in the J2000 coordinate system after rotation, or convert lab frame coordinates
to the J2000 coordinates after rotation.

Euler angles convention: Tait-Bryan angles, specifically, Z-Y-X, as
1. rotate lab frame around Z-axis of inertial frame by angle phi,
2. rotate lab frame around Y-axis of its own frame by angle theta,
3. rotate lab frame around X-axis of its own frame by angle psi.

Unit vector along the X-axis of the lab frame of the rigid body is also referred to as 
the axis-vector of the rigid body, while unit vector Z-axis of the lab frame of the rigid
body is referred to as its up-vector.

'''

## python 2 and 3 compatible
##

import numexpr as ne
from pymath.common import *


def fit_attitude(
        quat  = None,
        axis  = None,
        up    = None,
        phi   = None,
        theta = None,
        psi   = None):
    """Fit attitude of a rigid body.
There are three ways to specify the attitude of a rigid body:
  (1) quaternion,
  (2) axis and up vectors, or
  (3) Euler angles.
"""
    default_quat  = np.array([1.0, 0.0, 0.0, 0.0])
    default_axis  = np.array([1.0, 0.0, 0.0])
    default_up    = np.array([0.0, 0.0, 1.0])
    default_phi   = 0.0
    default_theta = 0.0
    default_psi   = 0.0
    if quat is None:
        if axis is None:
            if (phi is None) & (theta is None):
                axis = default_axis
                phi, theta, _ = xyz2ptr(*tuple(axis))
            else:
                if phi is None:
                    phi = default_phi
                if theta is None:
                    theta = default_theta
                axis = np.array(ptr2xyz(phi,theta,1.0))
            if up is None:
                if psi is None:
                    up = default_up
                    quat = from_axes(axis,up)
                    _, _, psi = angles(quat)
                else:
                    axis_phi,axis_theta,axis_rho = xyz2ptr(*tuple(axis))
                    quat = from_angles(axis_phi,axis_theta,psi)
                    up = rotate(quat=quat,vector=default_up)
            else:
                quat = from_axes(axis,up)
        else:
            if up is None:
                if psi is None:
                    psi = default_psi
                    up = default_up
                    quat = from_axes(axis,up)
                else:
                    axis_phi,axis_theta,axis_rho = xyz2ptr(*tuple(axis))
                    quat = from_angles(axis_phi,axis_theta,psi)
                    up = rotate(quat=quat,vector=default_up)
    else:
        axis = rotate(quat=quat,vector=default_axis)
        up = rotate(quat=quat,vector=default_up)
        psi = angles(quat)[2]
    axis = direction(axis)
    up = direction(up)
    return quat, axis, up, phi, theta, psi, {
        'quat':quat, 'axis':axis, 'up':up, 'phi':phi, 'theta':theta, 'psi':psi}

def axes2angles(axis,up):
    """Get Tait-Bryan angles (phi, theta, psi) from rotated axis-vector and up-vector.
"""
    phi,theta,rho = xyz2ptr(*tuple(axis))
    up0 = np.array([-1.0 * np.sin(theta)*np.cos(phi),\
        -1.0 * np.sin(theta)*np.sin(phi),\
        np.cos(theta)])
    sin_psi = norm(np.cross(up0,up,axis=0)) * \
        np.sign(np.sum(np.cross(up0,up,axis=0)*axis,axis=0))
    cos_psi = np.sum(up0*up,axis=0)
    psi = np.arctan2(sin_psi,cos_psi)
    return phi,theta,psi

def angles2axes(phi,theta,psi):
    """Get rotated axis-vector and up-vector from Tait-Bryan angles.
"""
    axis = ptr2xyz(phi,theta)
    up = np.array([np.sin(phi)*np.sin(psi) - np.sin(theta)*np.cos(phi)*np.cos(psi),\
        -np.cos(phi)*np.sin(psi)-np.sin(theta)*np.sin(phi)*np.cos(psi),\
        np.cos(theta)*np.cos(psi)])
    return axis,up

def matrix(q):
    """Convert quaternion to an equivalent rotation matrix.
"""
    #q_norm = np.sqrt(np.sum(np.power(q, 2.0)))
    #if q_norm <= DEPS:
        #return np.eye(3)
    #q = np.double(q) / q_norm
    return np.array([\
        [q[0]**2.0+q[1]**2.0-q[2]**2.0-q[3]**2.0, 2.0*(q[1]*q[2]-q[0]*q[3]), 2.0*(q[0]*q[2]+q[1]*q[3])],\
        [2.0*(q[0]*q[3]+q[1]*q[2]), q[0]**2.0-q[1]**2.0+q[2]**2.0-q[3]**2.0, 2.0*(q[2]*q[3]-q[0]*q[1])],\
        [2.0*(q[1]*q[3]-q[0]*q[2]), 2.0*(q[0]*q[1]+q[2]*q[3]), q[0]**2.0-q[1]**2.0-q[2]**2.0+q[3]**2.0]],dtype=q.dtype)
    #u = np.double(q[1:4])
    #sin_a = np.sqrt((u ** 2.0).sum())
    #if sin_a <= DEPS:
        #return np.eye(3)
    #cos_a = np.double(q[0])
    #phi = 2.0 * np.angle(cos_a + sin_a * 1.0j)
    #u = u / sin_a
    #if np.abs(phi) <= DEPS:
        #return np.eye(3)
    #else:
        #return rotmat(u, phi)

def conjugate(q):
    """Quaternion conjugate
"""
    return np.double([q[0], -1.0*q[1], -1.0*q[2], -1.0*q[3]])

def angles(quat):
    """Calculate Tait-Bryan angles from given quaternion.

The angles phi, theta and psi are specified as:
0. before rotation the lab frame is aligned to the inertial frame.
1. rotate the lab frame around the Z-axis of the inertial frame by phi.
2. rotate the lab frame around its own Y-axis by theta.
3. rotate the lab frame around its own X-axis by psi.

Therefore the Tait-Bryan angles are denoted as Z-Y-X.
"""
    axis = rotate(quat,[1,0,0])
    phi,theta,rho = xyz2ptr(*tuple(axis))
    up  = rotate(quat,[0,0,1])
    up0 = np.array([-1.0 * np.sin(theta)*np.cos(phi),\
        -1.0 * np.sin(theta)*np.sin(phi),\
        np.cos(theta)])
    sin_psi = norm(np.cross(up0,up,axis=0)) * \
        np.sign(np.sum(np.cross(up0,up,axis=0)*axis,axis=0))
    cos_psi = np.sum(up0*up,axis=0)
    psi = np.arctan2(sin_psi,cos_psi)
    return phi,theta,psi

def from_angles(phi,theta,psi=0.0):
    """Construct quaternion from Tait-Bryan angles phi, theta and psi.
"""
    if np.any(np.abs(psi)>DEPS):
        return np.array([\
            ne.evaluate("cos(phi*0.5) * cos(theta*0.5) * cos(psi*0.5) - sin(phi*0.5) * sin(theta*0.5) * sin(psi*0.5)"),\
            ne.evaluate("sin(phi*0.5) * sin(theta*0.5) * cos(psi*0.5) + cos(phi*0.5) * cos(theta*0.5) * sin(psi*0.5)"),\
            ne.evaluate("sin(phi*0.5) * cos(theta*0.5) * sin(psi*0.5) - cos(phi*0.5) * sin(theta*0.5) * cos(psi*0.5)"),\
            ne.evaluate("cos(phi*0.5) * sin(theta*0.5) * sin(psi*0.5) + sin(phi*0.5) * cos(theta*0.5) * cos(psi*0.5)")])
    else:
        return np.array([\
            ne.evaluate("cos(phi*0.5) * cos(theta*0.5)"),\
            ne.evaluate("sin(phi*0.5) * sin(theta*0.5)"),\
            ne.evaluate("-cos(phi*0.5) * sin(theta*0.5)"),\
            ne.evaluate("sin(phi*0.5) * cos(theta*0.5)")])

def from_angles_full(phi,theta,psi):
    """Construct quaternion from Tait-Bryan angles.
"""
    return np.array([\
        ne.evaluate("cos(phi*0.5) * cos(theta*0.5) * cos(psi*0.5) - sin(phi*0.5) * sin(theta*0.5) * sin(psi*0.5)"),\
        ne.evaluate("sin(phi*0.5) * sin(theta*0.5) * cos(psi*0.5) + cos(phi*0.5) * cos(theta*0.5) * sin(psi*0.5)"),\
        ne.evaluate("sin(phi*0.5) * cos(theta*0.5) * sin(psi*0.5) - cos(phi*0.5) * sin(theta*0.5) * cos(psi*0.5)"),\
        ne.evaluate("cos(phi*0.5) * sin(theta*0.5) * sin(psi*0.5) + sin(phi*0.5) * cos(theta*0.5) * cos(psi*0.5)")])

def multiply(q1, q2):
    """Multiplication of two quaternions q1 and q2.
"""
    d = {'q10':q1[0],'q11':q1[1],'q12':q1[2],'q13':q1[3],\
        'q20':q2[0],'q21':q2[1],'q22':q2[2],'q23':q2[3]}
    return np.array([ne.evaluate('q10*q20-q11*q21-q12*q22-q13*q23',local_dict=d),
        ne.evaluate('q10*q21+q20*q11+q12*q23-q22*q13',local_dict=d),
        ne.evaluate('q10*q22+q20*q12+q13*q21-q23*q11',local_dict=d),
        ne.evaluate('q10*q23+q20*q13+q11*q22-q21*q12',local_dict=d)])

def rotate(quat,vector):
    """Rotate input vector(s) with given quaternion(s)
"""
    d = {'q0':quat[0],'q1':quat[1],'q2':quat[2],'q3':quat[3],
         'r0':vector[0],'r1':vector[1],'r2':vector[2]}
    return np.array([ne.evaluate("(q0**2 + q1**2 - q2**2 - q3**2)*r0"+\
        "+2*(q1*q2 - q0*q3)*r1"+\
        "+2*(q0*q2 + q1*q3)*r2",local_dict=d),\
        ne.evaluate("(q0**2 - q1**2 + q2**2 - q3**2)*r1"+\
        "+2*(q0*q3 + q1*q2)*r0"+\
        "+2*(q2*q3 - q0*q1)*r2",local_dict=d),\
        ne.evaluate("(q0**2 - q1**2 - q2**2 + q3**2)*r2"+\
        "+2*(q0*q1 + q2*q3)*r1"+\
        "+2*(q1*q3 - q0*q2)*r0",local_dict=d)])

def from_matrix(R):
    """Construct quaternion from equivalent rotation matrix.
"""
    a = R[0,0]+R[1,1]+R[2,2]
    ux = (R[2,1] - R[1,2])*0.5
    uy = (R[0,2] - R[2,0])*0.5
    uz = (R[1,0] - R[0,1])*0.5
    b = ux**2 + uy**2 + uz**2
    sin_phi = np.sqrt(b)
    cos_phi = (a - 1.0)/2.0
    phi = np.angle(cos_phi + sin_phi * 1.0j)
    if np.abs(sin_phi) > DEPS:
        u = np.array([ux,uy,uz]) / sin_phi
        sin_phi_2 = np.sin(phi*0.5)
        return [np.cos(phi*0.5),
            sin_phi_2*u[0],
            sin_phi_2*u[1],
            sin_phi_2*u[2]]
    else:
        return [1,0,0,0]

def from_axes(axis,up):
    '''Find quaternion by axis and up vectors.

A rigid body is rotated to ax and up from [1,0,0] and [0,0,1]
respectively.
Axis and up are two intrinsic vectors of a rigid body.
They are perpendicular to each other.
'''
    phi,theta,rho = xyz2ptr(*tuple(axis))
    up0 = np.array([-1.0 * np.sin(theta)*np.cos(phi),\
        -1.0 * np.sin(theta)*np.sin(phi),\
        np.cos(theta)])
    sin_psi = norm(np.cross(up0,up,axis=0)) * \
        np.sign(np.sum(np.cross(up0,up,axis=0)*axis,axis=0))
    cos_psi = np.sum(up0*up,axis=0)
    psi = np.arctan2(sin_psi,cos_psi)
    return from_angles(phi,theta,psi)

def test(N=10000):
    '''Self-test routines.
'''
    phi,theta,psi = np.random.rand(3, N)
    phi = phi*2.0*np.pi
    theta = theta*np.pi - np.pi/2.0
    psi = psi*2.0*np.pi
    print(u'test from and to angles.')
    q = from_angles(phi,theta,psi)
    a,b,c = angles(q)
    print(np.allclose(np.sin(a), np.sin(phi)))
    print(np.allclose(np.cos(a), np.cos(phi)))
    print(np.allclose(np.sin(b), np.sin(theta)))
    print(np.allclose(np.cos(b), np.cos(theta)))
    print(np.allclose(np.sin(c), np.sin(psi)))
    print(np.allclose(np.cos(c), np.cos(psi)))
    print(u'test from and to axis.')
    axis = rotate(q,[1,0,0])
    up = rotate(q,[0,0,1])
    q1 = from_axes(axis,up)
    r = np.random.rand(3,1)
    print(np.allclose(rotate(q,r), rotate(q1,r)))
    q2 = from_axes(axis[:,0],up[:,0])
    print(np.allclose(rotate(q[:,0],r), rotate(q2,r)))
    print(u'test axes2angles.')
    print(np.allclose(rotate(from_angles(phi,theta,psi),r), rotate(from_angles(*tuple(axes2angles(axis,up))),r)))
    print(u'test angles2axes.')
    print(np.allclose(rotate(from_axes(axis,up),r), rotate(from_axes(*tuple(angles2axes(phi,theta,psi))),r)))
