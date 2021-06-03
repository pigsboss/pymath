"""Kernel matrix analysis using k-means clustering and NIPALS PCA.
Functions in this module are compatible with both Numpy arrays and
PyTables arrays.
"""

## python 2 and 3 compatible
##

from pymath.common import *
from time import time
from scipy.fftpack import fft,ifft
import scipy.sparse as sparse
import pymath.clfuns as clfuns
import pyopencl as cl
from pyopencl.array import vec as clvec
import tables
import numexpr as ne
import sys
from os import path
import multiprocessing as mp
from multiprocessing import Pool

current_path,_ = path.split(__file__)
with open(path.join(current_path,'kerana.c'),'r') as cl_src_file:
    cl_src = cl_src_file.read()

buffer_size = mp.cpu_count()*1*(1024**3)

def transpose(A_in, A_out):
    pass

def km2psf(H,obj_loc,obj_shape=None,obs_shape=None):
    """Get PSF at given location from kernel matrix.

H is the given kernel matrix.

obj_loc is the given location on object plane.

obj_shape is the shape of pixel grid on object plane specified as
(num_rows, num_cols).

obs_shape is the shape of pixel grid of image plane specified as
(num_rows, num_cols).

indexing as well as reshaping are performed from the last axis to the
first one ('C'-like order in numpy.reshape).
"""
    N, M = np.shape(H)
    loc_vector = np.array(obj_loc).ravel()
    ndims = np.size(loc_vector)
    if ndims == 2:
        if obj_shape is None:
            obj_shape = (np.uint64(np.sqrt(M)),np.uint64(np.sqrt(M)))
        if obs_shape is None:
            obs_shape = (np.uint64(np.sqrt(N)),np.uint64(np.sqrt(N)))
    else:
        if obj_shape is None:
            obj_shape = (M,1)
        if obs_shape is None:
            obs_shape = (N,1)
    shape_vector = np.array(obj_shape).ravel()
    k = 0
    for i in range(shape_vector.size):
        try:
            k += loc_vector[i]*np.prod(shape_vector[(i+1):])
        except:
            pass
    return np.reshape(H[:,k],obs_shape)

def dmvmod(A,Y,f):
    """Decomposed-matrix-vector modulation.

d = H x f
H is a N-by-M kernel matrix, f is a M-by-1 vector, d is modulated vector.
H is decomposed into sum_k(A_k x H_k).
A is a K-by-N array. The i-th row of A represents the diagonal of A_i.
Y is a K-by-M array. The i-th row of Y represents the first row vector of
H_i.
"""
    K = len(A)
    N,M = A[0].shape
    size_f = np.size(f)
    if size_f != M:
        raise StandardError('Shape of matrix or vector is wrong.')
    print(u'Number of clusters: %d'%K)
    print(u'Original kernel matrix: %d x %d'%(N,M))
    d = np.zeros(N, dtype=f.dtype)
    F = fft(f)
    for k in range(K):
        d += A[k].dot(np.real(ifft(np.conj(fft(Y[k]))*F)))
    return d

def dmvcorr(A,Y,d):
    """Decomposed-matrix-vector correlation transform.

c = Ht x d
Ht is a M-by-N matrix, the transpose of kernel matrix H.
d is a N-by-1 vector, c is transformed vector.
H is decomposed into sum_k(A_k x H_k).
A is a K-by-N array. The i-th row of A represents the diagonal of A_i.
Y is a K-by-M array. The i-th row of Y represents the first row vector of
H_i.
"""
    K,M = np.shape(Y)
    size_d = np.size(d)
    num_rows_A,N = np.shape(A)
    if size_d != N:
        raise StandardError('Shape of matrix or vector is wrong.')
    if num_rows_A != K:
        raise StandardError('A and Y must contain the same number of rows.')
    print(u'Number of clusters: %d'%K)
    print(u'Original kernel matrix: %d x %d'%(N,M))
    c = np.zeros(M,dtype=d.dtype)
    if M > N:
        for k in range(K):
            c += np.real(ifft(fft(Y[k]) *\
                fft(np.pad(A[k]*d,(0,M-N),mode='wrap'))))
    elif M < N:
        for k in range(K):
            c += np.real(ifft(fft(Y[k]) * fft(A[k,0:M]*d[0:M])))
    else:
        for k in range(K):
            c += np.real(ifft(fft(Y[k]) * fft(A[k]*d)))
    return c

def dmacorr(A,Y,m=None,P=None):
    """Decomposed-matrix-vector auto correlation.

P = Ht x Ma x H
Ht is a M-by-N matrix, the transpose of kernel matrix H.
Ma is a N-by-N diagonal matrix, m is its diagonal.
P  is a M-by-M matrix.
H is decomposed into sum_k(A_k x H_k).
A is a K-by-N array. The i-th row of A represents the diagonal of A_i.
Y is a K-by-M array. The i-th row of Y represents the first row vector of
H_i.
"""
    K,M = np.shape(Y)
    num_rows_A,N = np.shape(A)
    if m is None:
        print(u'No mask vector specified.')
        m = np.ones(N, dtype=Y.dtype)
    size_m = np.size(m)
    if size_m != N:
        raise StandardError('Shape of matrix or vector is wrong.')
    if num_rows_A != K:
        raise StandardError('A and Y must contain the same number of rows.')
    print(u'Number of clusters: %d'%K)
    print(u'Original kernel matrix: %d x %d'%(N,M))
    if P is None:
        print(u'No output matrix specified.')
        P = np.zeros((M,M), dtype=Y.dtype)
    if M > N:
        for k in range(K):
            f = Y[k]
            F = fft(f)
            r = np.roll(np.flipud(f),1)
            for i in range(M):
                P[:,i] += np.real(ifft(F *\
                    fft(np.pad(A[k]*m,(0,M-N),mode='wrap')*np.roll(r,i))))
    elif M < N:
        for k in range(K):
            f = Y[k]
            F = fft(f)
            r = np.roll(np.flipud(f),1)
            for i in range(M):
                P[:,i] += np.real(ifft(F*fft(A[k,0:M]*m[0:M]*np.roll(r,i))))
    else:
        for k in range(K):
            f = Y[k]
            F = fft(f)
            r = np.roll(np.flipud(f),1)
            for i in range(M):
                print(u"%d, %d"%(i,k))
                P[:,i] += np.real(ifft(F*fft(A[k]*m*np.roll(r,i))))
    return P

def circshift(A_in, A_out, initial=0, step=1, shift=None, obj_shape=None, obs_shape=None, order=None):
    """Circularly shift each row of input matrix A.

The default is to shift each row leftward.

initial is the number of elements by which the first row is shifted.

step is the increment of the quantity of each shifting increases from
the first row to the last.

shift specifies leftward offset for each row vector.

obj_shape and obs_shape are shapes of the object and the observed data of the original
modulation equation.

order is used to ravel N-D array into 1-D array or reshape the 1-D
array back to the original N-D array.
"""
    if A_in is A_out:
        print(u'circular-shift will be performed in-place.')
    num_rows_in, num_cols_in = np.shape(A_in)
    num_rows_out, num_cols_out = np.shape(A_out)
    if (num_cols_in != num_cols_out) or (num_rows_in != num_rows_out):
        raise StandardError('Shapes of A_in and A_out are different.')
    if obj_shape is None:
        obj_shape = (num_cols_in,)
    if obs_shape is None:
        obs_shape = (num_rows_in,)
    fdim = len(obj_shape)
    ddim = len(obs_shape)
    print 'Object is %d-D while observed data is %d-D.'%(fdim,ddim)
    tic = time()
    if ddim == 1:
        if shift is None:
            shift = np.int64(np.arange(initial, initial+num_rows_in*step, step))
        for i in range(num_rows_in):
            A_out[i,:] = np.roll(A_in[i,:], -shift[i])
            if np.mod(i+1,512) == 0:
                sys.stdout.write('\r%d/%d rows shifted, %d seconds elapsed, %d seconds left.'\
                    %(i+1,num_rows_in,time()-tic,(time()-tic)/(i+1.0)*(num_rows_in-i-1.0)))
                sys.stdout.flush()
    elif ddim == 2:
        if shift is None:
            u,v = np.meshgrid(range(obs_shape[0]), range(obs_shape[1]), indexing='xy')
            shift0 = np.ravel(v, order=order)
            shift1 = np.ravel(u, order=order)
        else:
            shift0 = shift[0]
            shift1 = shift[1]
        for i in range(num_rows_in):
            p = np.reshape(A_in[i,:], obj_shape, order=order)
            p = np.roll(np.roll(p, -shift0[i], axis=0), -shift1[i], axis=1)
            A_out[i,:] = np.ravel(p,order=order)
            if np.mod(i+1,512) == 0:
                sys.stdout.write('\r%d/%d rows shifted, %d seconds elapsed, %d seconds left.'\
                    %(i+1,num_rows_in,time()-tic,(time()-tic)/(i+1.0)*(num_rows_in-i-1.0)))
                sys.stdout.flush()
    else:
        raise StandardError('%d-D observed data is not supported yet.'%ddim)
    print '\nComplete.'

def pcanipals(X,R=None,P=None,T=None,J=None,eps=1e-4,max_loop=1000,progress=512):
    """PCA using NIPALS algorithm.

X is the input N x M matrix.

R is the residual matrix, default is None. Shape of R must be the same as
X. If specified, R must be either taken from a previous run of this or
copied from X.

P is the loading matrix, default is None. Number of rows of P must be
exactly M and number of columns of P must be at least J.

T is PC scores, default is None. Number of rows of T must be exactly N.
Number of columns of T must be at least J.

J is the expected number of principal components (PCs). Default is the
number of columns of X.

eps is maximum accepted relative error used to stop the iteration. Default
is calculated according to covariance and scale of residual.

progress is number of steps before the next progress update in the command line
environment. Set progress to -1 to disables this function.

Returns:
R, P, T, and eps_list (convergence).
"""
    # parse user input.
    X,R,P,T,J,eps,max_loop,progress,j,M,N = \
        parse_pcanipals_inputs(X,R,P,T,J,eps,max_loop,progress)
    if j >= J:
        print(u'All %d scores have been calculated.'%J)
        return R,P,T

    # implement NIPALS-PCA algorithm.
    Pcv = np.empty(M, dtype=X.dtype)
    eps_list = np.zeros(max_loop)
    tic = time()
    nbcols = buffer_size/X.dtype.itemsize/N
    while j < J:
        eigen = 0.0
        eigen_old = 100.0+eps
        Tcv = R[:,j]
        t = 0
        while np.abs(eigen - eigen_old) > eps*0.5*np.abs(eigen + eigen_old):
            i = 0
            while i<M:
                n = min(nbcols, M-i)
                Pcv[i:i+n] = np.dot(R[:,i:i+n], Tcv)
                i += n
                if progress>0:
                    sys.stdout.write(u"\r%d/%d loadings updated."%(i,M))
                    sys.stdout.flush()
            P_norm = np.linalg.norm(Pcv)
            if P_norm <= np.finfo(X.dtype).eps:
                print(u'\nAll PCs have been calculated.')
                break
            Pcv = Pcv / P_norm
            i = 0
            while i<M:
                n = min(nbcols, M-i)
                Tcv = np.matmul(R[:,i:i+n], Pcv[i:i+n])
                i += n
                if progress>0:
                    sys.stdout.write(u"\r%d/%d PC score elements updated."%(i,M))
                    sys.stdout.flush()
            eigen_old = eigen
            eigen = np.linalg.norm(Tcv)
            eps_list[t] = eigen - eigen_old
            if t >= max_loop:
                print(u'\nMax number of loops reached.')
                break
            t += 1
            print(u' Residual: %f'\)
                %(np.abs(eigen - eigen_old)*2.0/np.abs(eigen + eigen_old))
        P[:,j] = Pcv
        T[:,j] = Tcv
        i = 0
        while i<M:
            n = min(nbcols, M-i)
            R[:,i:i+n] = R[:,i:i+n] - np.matmul(np.reshape(T[:,j],(-1,1)), np.reshape(P[i:i+n,j],(1,-1)))
            i += n
            if progress>0:
                sys.stdout.write(u"\r%d/%d residual columns updated."%(i,M))
                sys.stdout.flush()
        j += 1
        print(u"\n%d/%d scores calculated, %d seconds elapsed, %d seconds left."%(j,J,time()-tic,(time()-tic)*1.0/j*(J-j)))
    print(u'\nComplete.')
    return R,P,T,eps_list


def hpm_pcanipals(X,R=None,P=None,T=None,J=None,eps=1e-4,max_loop=1000,progress=512):
    """PCA using NIPALS algorithm with huge physics memory.

X is the input N x M matrix.

R is the residual matrix, default is None. Shape of R must be the same as
X. If specified, R must be either taken from a previous run of this or
copied from X.

P is the loading matrix, default is None. Number of rows of P must be
exactly M and number of columns of P must be at least J.

T is PC scores, default is None. Number of rows of T must be exactly N.
Number of columns of T must be at least J.

J is the expected number of principal components (PCs). Default is the
number of columns of X.

eps is maximum accepted relative error used to stop the iteration. Default
is calculated according to covariance and scale of residual.

progress is number of steps before the next progress update in the command line
environment. Set progress to -1 to disables this function.

Returns:
R, P, T, and eps_list (convergence).
"""
    # parse user input.
    X,R,P,T,J,eps,max_loop,progress,j,M,N = \
        parse_pcanipals_inputs(X,R,P,T,J,eps,max_loop,progress,in_memory=True)
    if j >= J:
        print(u'All %d scores have been calculated.'%J)
        return R,P,T

    # implement NIPALS-PCA algorithm.
    Tcv = np.empty(N, dtype=X.dtype)
    Pcv = np.empty(M, dtype=X.dtype)
    eps_list = np.zeros(max_loop)
    tic = time()
    while j < J:
        eigen = 0.0
        eigen_old = 100.0+eps
        Tcv = R[:,j]
        t = 0
        while np.abs(eigen - eigen_old) > eps*0.5*np.abs(eigen + eigen_old):
            Pcv    = np.ravel(np.matmul(np.reshape(Tcv, (1,-1)), R))
            P_norm = np.sqrt(np.sum(Pcv**2.0))
            if P_norm <= np.finfo(X.dtype).eps:
                print(u'\nAll PCs have been calculated.')
                break
            Pcv = Pcv / P_norm
            Tcv = np.ravel(np.matmul(R, np.reshape(Pcv, (-1,1))))
            eigen_old = eigen
            eigen = np.sqrt(np.sum(Tcv**2.0))
            eps_list[t] = eigen - eigen_old
            if t >= max_loop:
                print(u'\nMax number of loops reached.')
                break
            t += 1
            print(u' Residual: %f'%(np.abs(eigen - eigen_old)*2.0/np.abs(eigen + eigen_old)))
        P[:,j] = Pcv
        T[:,j] = Tcv
        R -= np.matmul(np.reshape(T[:,j],(-1,1)),np.reshape(P[:,j],(1,-1)))
        j += 1
        print(u"\n%d/%d scores calculated, %d seconds elapsed, %d seconds left."%(j,J,time()-tic,(time()-tic)*1.0/j*(J-j)))
    print(u'\nComplete.')
    return R,P,T,eps_list


def cl_pcanipals(X,R=None,P=None,T=None,J=None,eps=1e-4,max_loop=1000,\
        progress=1024,dev_id=None):
    """PCA using NIPALS algorithm implemented with OpenCL.
    X is the input N x M matrix.
    R is the residual matrix, default is None. Shape of R must be the same as
    X. If specified, R must be either taken from a previous run of this or
    copied from X.
    P is the loading matrix, default is None. Number of rows of P must be
    exactly M and number of columns of P must be at least J.
    T is PC scores, default is None. Number of rows of T must be exactly N.
    Number of columns of T must be at least J.
    J is the expected number of principal components (PCs). Default is the
    number of columns of X.
    eps is maximum accepted relative error used to stop the iteration. Default
    is calculated according to covariance and scale of residual.
    progress is number of steps before the next progress update in the command line
    environment. Set progress to -1 to disables this function.
    """
    # parse user input.
    X,R,P,T,J,eps,max_loop,progress,j,M,N = \
        parse_pcanipals_inputs(X,R,P,T,J,eps,max_loop,progress)
    if j >= J:
        print(u'All %d scores have been calculated.'%J)
        return R,P,T

    # prepare OpenCL
    if dev_id is None:
        device = clfuns.default_device()
    else:
        device = clfuns.list_devices()[dev_id]
    compute_units = device.max_compute_units
    print(u'\n==================================================')
    print(u'OpenCL Platform: %s'%device.platform.name)
    print(u'Device name: %s'%device.name)
    if device.type == cl.device_type.GPU:
        print(u'Device type: GPU')
        ws = 64
        global_work_size = compute_units * 7 * ws
        while ((N/4) > global_work_size) and \
            (((N/4) % global_work_size) != 0):
            global_work_size += ws
        if (N/4) < global_work_size:
            global_work_size = (N/4)
        local_work_size = ws
        is_cpu = 0
    else:
        print(u'Device type: CPU')
        global_work_size = compute_units * 1
        local_work_size = 1
        is_cpu = 1
    print(u'Compute Units: %d'%compute_units)
    print(u'Global work size: %d'%global_work_size)
    print(u'Local work size: %d'%local_work_size)
    num_groups = global_work_size / local_work_size
    print(u'Number of groups: %d'%num_groups)
    print(u'\n==================================================')

    ctx = cl.Context([device])
    mf = cl.mem_flags
    queue = cl.CommandQueue(ctx)
    opt_str = "-DIS_CPU=%d -DNITEMS=%d"%(is_cpu, N)
    prg = cl.Program(ctx, cl_src).build(options=opt_str)

    Rcv_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.ALLOC_HOST_PTR, size = N*4)
    Tcv_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.ALLOC_HOST_PTR, size = N*4)
    Tcv_next_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.ALLOC_HOST_PTR, size = N*4)
    Pcv_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.ALLOC_HOST_PTR, size = M*4)
    gsum_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.ALLOC_HOST_PTR,\
        size = num_groups*4)
    upl = cl.Kernel(prg, "update_pc_loading_local_fp32")
    upg = cl.Kernel(prg, "update_pc_loading_global_fp32")
    upsc = cl.Kernel(prg, "update_pc_score_fp32")
    upl.set_args(Rcv_buf, Tcv_buf, gsum_buf, cl.LocalMemory(local_work_size*4))
    upg.set_arg(0, gsum_buf)
    upg.set_arg(1, Pcv_buf)
    upsc.set_arg(0, Rcv_buf)
    upsc.set_arg(1, Tcv_next_buf)
    upsc.set_arg(2, Pcv_buf)

    Tcv = np.empty(N, dtype=np.float32)
    Pcv = np.empty(M, dtype=np.float32)

    # implement GPU-NIPALS-PCA algorithm.
    eps_list = np.zeros(max_loop)
    tic = time()
    while j < J:
        eigen = 0.0
        eigen_old = 100.0+eps
        t = 0
        Tcv[:] = R[:,j]
        while np.abs(eigen - eigen_old) > eps*0.5*np.abs(eigen + eigen_old):
            cl.enqueue_copy(queue, Tcv_buf, Tcv)
            cl.enqueue_copy(queue, Tcv_next_buf, np.zeros(N, dtype=np.float32))
            for i in range(M):
                # OpenCL implementation
                cl.enqueue_copy(queue, Rcv_buf, np.float32(R[:,i]))
                upg.set_arg(2, np.uint32(i))
                upsc.set_arg(3, np.uint32(i))
                exec_evt = cl.enqueue_nd_range_kernel(queue,upl,\
                    (global_work_size,),(local_work_size,))
                exec_evt.wait()
                exec_evt = cl.enqueue_nd_range_kernel(queue,upg,\
                    (global_work_size,),(local_work_size,))
                exec_evt.wait()
                exec_evt = cl.enqueue_nd_range_kernel(queue,upsc,\
                    (global_work_size,),(local_work_size,))
                exec_evt.wait()

                if progress>0 and np.mod(i+1, progress)==0:
                    sys.stdout.write(u"\r%d/%d PC elements updated."%(i+1,M))
                    sys.stdout.flush()

            cl.enqueue_read_buffer(queue, Tcv_next_buf, Tcv).wait()
            cl.enqueue_read_buffer(queue, Pcv_buf, Pcv).wait()

            P_norm = np.sqrt(np.sum(Pcv**2.0))
            if P_norm <= np.finfo(X.dtype).eps:
                print(u'\nAll PCs have been calculated.')
                break

            Pcv[:] = Pcv / P_norm
            Tcv[:] = Tcv / P_norm

            eigen_old = eigen
            eigen = np.sqrt(np.sum(Tcv**2.0))
            eps_list[t] = eigen - eigen_old
            if t >= max_loop:
                print(u'\nMax number of loops reached.')
                break
            t += 1
            print(u' Residual: %f'\)
                %(np.abs(eigen - eigen_old)*2.0/np.abs(eigen + eigen_old))

        P[:,j] = Pcv
        T[:,j] = Tcv

        for i in range(M):
            R[:,i] = R[:,i] - T[:,j]*P[i,j]
            if progress>0 and np.mod(i+1,progress)==0:
                sys.stdout.write(u"\r%d/%d residual columns updated."%(i+1,M))
                sys.stdout.flush()
        j += 1
        print(u"\n%d/%d scores calculated, %d seconds elapsed, %d seconds left."%(j,J,time()-tic,(time()-tic)*1.0/j*(J-j)))
    print(u'\nComplete.')
    return R,P,T,eps_list

def parse_pcanipals_inputs(X,R=None,P=None,T=None,J=None,eps=1e-4,max_loop=1000,progress=512,in_memory=False):
    """Parse user inputs for pcanipals and gpupcanipals.

"""
    num_rows_X, num_cols_X = np.shape(X)
    N = num_rows_X
    M = num_cols_X
    print(u'Shape of the input matrix: %d x %d'%(num_rows_X, num_cols_X))
    if J is None:
        print(u'No expected number of principal components given.')
        if P is None:
            print(u'No hint from shape of loading matrix. Use default.')
            J = M
        else:
            _, num_cols_P = np.shape(P)
            print(u'Loading matrix contains %d columns.'%num_cols_P)
            print(u'Use %d as expected number of principal components.'%num_cols_P)
            J = num_rows_P
    if R is None:
        print(u'No residual matrix given. Create numpy array by default.')
        R = X.copy()
    elif R is X:
        print(u'User prefer in-place PCA on the input matrix.')
        print(u'The original input matrix will be altered.')
    else:
        num_rows_R, num_cols_R = np.shape(R)
        print(u'Shape of the residual: %d x %d'%(num_rows_R, num_cols_R))
        if (num_rows_R != N) or (num_cols_R != M):
            raise StandardError('Shape of residual matrix is wrong.')
    if P is None:
        print(u'No loading matrix given. Create numpy array by default.')
        P = np.zeros(((M,J)), dtype=X.dtype)
    else:
        num_rows_P, num_cols_P = np.shape(P)
        print(u'Shape of the loading matrix: %d x %d'%(num_rows_P, num_cols_P))
        if num_rows_P != M:
            raise StandardError('Shape of loading matrix is wrong.')
        if num_cols_P < J:
            raise StandardError('Insufficient number of columns in loading matrix.')
    if T is None:
        print(u'No score matrix given. Create numpy array by default.')
        T = np.zeros((N,J), dtype=X.dtype)
    else:
        num_rows_T, num_cols_T = np.shape(T)
        print(u'Shape of the score matrix: %d x %d'%(num_rows_T, num_cols_T))
        if num_rows_T != N:
            raise StandardError('Shape of score matrix is wrong.')
        if num_cols_T < J:
            raise StandardError('Insufficient number of columns in score matrix.')
    if eps is None:
        print(u'No eps given.')
        eps = 1.732* N**0.45 * np.finfo(X.dtype).eps
        print(u'Set eps to %g'%eps)
    j = 0
    # determine if this is a new run or resumed from a previous one.
    if np.std(T[:,j])<=np.finfo(X.dtype).eps:
        print(u'Score matrix is empty.')
        print(u'This is a new run on the input matrix.')
        if in_memory:
            x0 = np.sum(R,axis=1)
            x0 = x0 / np.double(M)
            R -= np.repeat(np.reshape(x0,(-1,1)), M, axis=1)
        else:
            x0 = np.zeros(N,dtype=X.dtype)
            nbcols = buffer_size / x0.dtype.itemsize / N
            i = 0
            while i<M:
                n   = min(nbcols, M-i)
                x0 += np.sum(R[:,i:i+n], axis=1)
                i  += n
                if progress>0:
                    sys.stdout.write(u"\r%d/%d residual columns averaged."%(i,M))
                    sys.stdout.flush()
            x0 = x0 / np.double(M)
            x0 = x0.reshape((N,1))
            i = 0
            while i<M:
                n = min(nbcols, M-i)
                R[:,i:i+n] = R[:,i:i+n] - x0
                i += n
                if progress>0:
                    sys.stdout.write(u"\r%d/%d residual columns updated."%(i,M))
                    sys.stdout.flush()
    else:
        print(u'PC score %d is calculated.'%(j+1))
        print(u'Score matrix is not empty.')
        print(u'Try to find the last breakpoint.')
        j += 1
        while j < J:
            if np.std(T[:,j])>np.finfo(X.dtype).eps:
                print(u'PC score %d is calculated.'%(j+1))
                j += 1
            else:
                print(u'PC score %d is not calculated yet.'%(j+1))
                break
        print(u'%d scores have been calculated.'%j)
    print(u'\nCalculate PC score %d to %d.'%(j+1,J))
    return X,R,P,T,J,eps,max_loop,progress,j,M,N

def kmeans(X,K=None,C=None,S=None,D=None,max_loop=100):
    """K-means cluster analysis.

"""
    X,K,C,S,D,max_loop,M,N,C_norm = parse_kmeans_inputs(X,K,C,S,D,max_loop)

    # implement k-means algorithm
    delta = np.ones(K,dtype=C.dtype)
    l = 0
    tic = time()
    while np.any(delta >= np.finfo(C.dtype).eps):
        # assignment step
        S_old = np.copy(S)
        for i in range(N):
            for k in range(K):
                D[k,i] = np.sqrt(np.sum((X[i]-C[k])**2.0))
            S[i] = np.argmin(D[:,i])

        # update step
        L = slots2lists(S,K)
        for k in range(K):
            if np.size(L[k]) > 0:
                C_old = np.copy(C[k])
                C[k] = 0
                for i in L[k]:
                    C[k] += X[i]
                C[k] = C[k] / np.double(np.size(L[k]))
                delta[k] = np.linalg.norm(C_old - C[k])
            else:
                delta[k] = 0
        #print delta
        l += 1
        sys.stdout.write(u'\rloop %d, %d/%d clusters converged, %d seconds elapsed.'\
            %(l,np.sum(delta < np.finfo(C.dtype).eps),K,time()-tic))
        sys.stdout.flush()
        if l>= max_loop:
            print(u'\nMax number of loops reached.')
            break
    print(u'\nComplete.')
    return S,C

def __kmeans_assign(pid, nps, X, C, S, N, M, K):
    for i in range(pid, N, nps):
        S[i] = np.argmin(
            np.linalg.norm(
                np.repeat(np.reshape(X[(i*M):((i+1)*M)],(1,-1)), K, axis=0) - np.reshape(C,(K,M)),
                axis=1)
        )

def __kmeans_update(pid, nps, X, C, delta, L, N, M, K):
    for k in range(pid, K, nps):
        if np.size(L[k]) > 0:
            C_old = np.copy(C[(k*M):((k+1)*M)])
            C_tmp = np.zeros(M)
            for i in L[k]:
                C_tmp += X[(i*M):((i+1)*M)]
            C[(k*M):((k+1)*M)] = C_tmp/np.double(np.size(L[k]))
            delta[k] = np.linalg.norm(C_old - C[(k*M):((k+1)*M)])
        else:
            delta[k] = 0

def mp_kmeans(X,K=None,C=None,S=None,D=None,max_loop=100):
    """K-means cluster analysis with multiple processing.

"""
    X,K,C,S,D,max_loop,M,N,C_norm = parse_kmeans_inputs(X,K,C,S,D,max_loop)

    # implement k-means algorithm
    delta = np.ones(K,dtype=C.dtype)

    ncpus = mp.cpu_count()
    nps   = min(N, ncpus)
    X_mp  = mp.Array('d', np.ravel(X), lock=False)
    C_mp  = mp.Array('d', np.ravel(C), lock=False)
    S_mp  = mp.Array('i', S, lock=False)
    d_mp  = mp.Array('d', delta, lock=False)
    
    l = 0
    tic = time()
    while np.any(delta >= np.finfo(C.dtype).eps):
        # assignment step
        ps = []
        for i in range(nps):
            p = mp.Process(target=__kmeans_assign, args=(i,nps,X_mp,C_mp,S_mp,N,M,K))
            p.start()
            ps.append(p)
        for p in ps:
            p.join()
        # update step
        L = slots2lists(np.array(S_mp),K)
        ps = []
        for i in range(nps):
            p = mp.Process(target=__kmeans_update, args=(i,nps,X_mp,C_mp,d_mp,L,N,M,K))
            p.start()
            ps.append(p)
        for p in ps:
            p.join()
        delta = np.array(d_mp)
        #print delta
        l += 1
        sys.stdout.write(u'\rloop %d, %d/%d clusters converged, %d seconds elapsed.'\
            %(l,np.sum(delta < np.finfo(C.dtype).eps),K,time()-tic))
        sys.stdout.flush()
        if l>= max_loop:
            print(u'\nMax number of loops reached.')
            break
    print(u'\nComplete.')
    S = np.array(S_mp)
    C = np.array(C_mp)
    return S,C

def cl_kmeans(X,K=None,C=None,S=None,D=None,max_loop=100,dev_id=None):
    """K-means cluster analysis.

"""
    X,K,C,S,D,max_loop,M,N,C_norm = parse_kmeans_inputs(X,K,C,S,D,max_loop)

    # prepare OpenCL
    if dev_id is None:
        device = clfuns.default_device()
    else:
        device = clfuns.list_devices()[dev_id]
    compute_units = device.max_compute_units
    print(u'\n==================================================')
    print(u'OpenCL Platform: %s'%device.platform.name)
    print(u'Device name: %s'%device.name)
    if device.type == cl.device_type.GPU:
        print(u'Device type: GPU')
        ws = 64
        global_work_size = compute_units * 7 * ws
        while ((M/4) > global_work_size) and \
            (((M/4) % global_work_size) != 0):
            global_work_size += ws
        if (M/4) < global_work_size:
            global_work_size = (M/4)
        local_work_size = ws
        is_cpu = 0
    else:
        print(u'Device type: CPU')
        global_work_size = compute_units * 1
        local_work_size = 1
        is_cpu = 1
    print(u'Compute Units: %d'%compute_units)
    print(u'Global work size: %d'%global_work_size)
    print(u'Local work size: %d'%local_work_size)
    num_groups = global_work_size / local_work_size
    print(u'Number of groups: %d'%num_groups
    print(u'\n==================================================')

    ctx = cl.Context([device])
    mf = cl.mem_flags
    queue = cl.CommandQueue(ctx)

    # implement k-means algorithm with OpenCL
    X_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.ALLOC_HOST_PTR, size=M*4)
    C_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.ALLOC_HOST_PTR, size=M*4)
    D_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=num_groups*4)
    gd2 = np.empty(num_groups, dtype=np.float32)
    opt_str = "-DIS_CPU=%d -DNITEMS=%d"%(is_cpu, M)
    prg = cl.Program(ctx, cl_src).build(options=opt_str)
    dist = cl.Kernel(prg, "dist_local_fp32")
    dist.set_args(X_buf, C_buf, D_buf, cl.LocalMemory(local_work_size*4))
    delta = np.ones(K,dtype=C.dtype)
    l = 0
    tic = time()
    while np.any(delta >= np.finfo(C.dtype).eps):
        # assignment step
        S_old = np.copy(S)
        for i in range(N):
            cl.enqueue_copy(queue, X_buf, np.float32(X[i]))
            for k in range(K):
                cl.enqueue_copy(queue, C_buf, np.float32(C[k]))
                exec_evt = cl.enqueue_nd_range_kernel(queue,dist,\
                    (global_work_size,),(local_work_size,))
                exec_evt.wait()
                cl.enqueue_read_buffer(queue, D_buf, gd2)
                D[k,i] = np.sqrt(np.sum(gd2))
                #D[k,i] = np.sqrt(np.sum((X[i]-C[k])**2.0))
                #print '%f, %f'%(Dki, D[k,i])
            S[i] = np.argmin(D[:,i])

        # update step
        L = slots2lists(S,K)
        for k in range(K):
            C_old = np.copy(C[k])
            C[k] = 0
            for i in L[k]:
                C[k] += X[i]
            C[k] = C[k] / np.double(np.size(L[k]))
            C_norm[k] = np.sqrt(np.sum(C[k]**2.0))
            delta[k] = np.sqrt(np.sum((C_old - C[k])**2.0))

        #print delta
        l += 1
        sys.stdout.write(u'\rloop %d, %d/%d clusters converged, %d seconds elapsed.'\
            %(l,np.sum(delta < np.finfo(C.dtype).eps),K,time()-tic))
        sys.stdout.flush()
        if l>= max_loop:
            print(u'\nMax number of loops reached.')
            break
    print(u'\nComplete.')
    return S,C

def parse_kmeans_inputs(X,K=None,C=None,S=None,D=None,max_loop=100):
    # decide expected number of clusters from user input
    N,M = np.shape(X)
    if K is None:
        print(u"No explicit expected number of clusters.")
        if C is not None:
            K = np.shape(C)[0]
            print(u"User input central vectors suggests %d clusters expected."%K)
        else:
            K = np.uint64(np.sqrt(np.double(N)))
            print(u"No implied expected number of clusters.")
            print(u"%d clusters expected by default."%K)
    # check user input central vectors
    if C is None:
        print(u"No central vectors input.")
        C = np.zeros((K,M),dtype=X.dtype)
    else:
        num_rows_C, num_cols_C = np.shape(C)
        if (num_rows_C != K) or (num_cols_C != M):
            raise StandardError('Shape of input central vector array is wrong.')
    # check user input vector slots
    if S is None:
        print(u"No cluster slots input.")
        S = np.empty(N, dtype='uint64')
    # check user input distance matrix
    if D is None:
        print(u"No distance matrix input.")
        D = np.empty((K,N),dtype=X.dtype)
    # initialize cluster lists
    #L = [[]]*K
    C_norm = np.linalg.norm(C, axis=1)
    if np.all(C_norm <= np.finfo(C.dtype).eps):
        print(u'Central vectors are not initialized.')
        # randomly initialize clusters
        idx = np.random.choice(N,size=K,replace=False)
        for k in range(K):
            C[k] = X[idx[k]]
    return X,K,C,S,D,max_loop,M,N,C_norm

def circdecomp(X,S,A=None,Y=None,shift=None,obj_shape=None,sparse_matrix=False):
    """Circular-shift decomposition

Decompose kernel matrix X into a series of circulant matrices multiplied
by diagonal matrices. The circulant vector of each circulant matrix and the
diagonal vector of each diagonal matrix are returned.

X is the input kernel matrix.

S is slots matrix calculated by k-means cluster analysis.

shift specifies leftward offset for each row vector of X, as in circshift.

The cluster analysis is performed on the sparse representation of X, i.e.,
the PC score matrix of X.
"""
    K = np.uint64(np.max(S)+1)
    N, M = np.shape(X)
    if shift is None:
        shift = np.arange(N)
    try:
        shift0, shift1 = shift
        shift0 = np.mod(shift0, obj_shape[0])
        shift1 = np.mod(shift1, obj_shape[1])
        shift = shift0*obj_shape[1]+shift1
    except:
        pass
    if A is None:
        print(u'No diagonal vector matrix input.')
        if sparse_matrix:
            print(u'Create sparse diagonal vector matrix.')
            A = [sparse.dok_matrix((N,M), dtype=X.dtype) for k in range(K)]
        else:
            print(u'Create dense diagonal vector matrix.')
            A = np.zeros((K,N,M), dtype=X.dtype)
    if Y is None:
        print(u'No circulant vector matrix input.')
        Y = np.empty((K,M), dtype=X.dtype)
    for k in range(K):
        Y[k] = 0.0
        t = 0.0
        for i in np.argwhere(S[:]==k).ravel():
            A[k][i,shift[i]] = 1
            Y[k] += X[i]
            t    += 1.0
        if t>0:
            Y[k] = Y[k] / t
        else:
            print(u'Cluster %d is empty.'%k)
    return A,Y

def slots2lists(S,K):
    L = [[] for i in range(K)]
    N = np.size(S)
    for i in range(N):
        t = S[i]
        if t >= 0:
            L[t].append(i)
    return L

def binormal(shape=None,mu=(0.0,0.0),sigma=(1.0,1.0),rho=0.0):
    """Generate binormal PDF on mesh grid of given shape.

"""
    if np.isscalar(shape):
        shape = (shape, shape)
    if np.isscalar(mu):
        mu = (mu, mu)
    if np.isscalar(sigma):
        sigma = (sigma, sigma)
    x = np.arange(shape[1]) - (shape[1]-1)*0.5
    y = np.arange(shape[0]) - (shape[0]-1)*0.5
    x,y = np.meshgrid(x,y)
    g = np.exp(-0.5/np.sqrt(1.0-rho**2.0)*((x-mu[0])**2.0/sigma[0]**2.0 +\
        (y-mu[1])**2.0/sigma[1]**2.0 - \
        2.0*rho*(x-mu[0])*(y-mu[1])/sigma[0]/sigma[1]))
    g = g / np.sum(g)
    return g

def ellipse2covariance(u,a,ecc):
    """Compute covariance from error ellipse.

Calculate covariance (represented by variances and correlation coefficient)
from prediction ellipse parameters u, a and ecc, where
u is the direction of the major axis of the ellipse, a is the length of the
major axis, and ecc is the eccentricity of the ellipse.
    """
    u = np.array(u) / np.sqrt(np.sum(np.array(u)**2.0))
    b = a*np.sqrt(1.0-ecc**2.0)
    ua = np.array(u)
    ub = np.array((-1.0*ua[1],ua[0]))
    U = np.matrix(((ua[0],ub[0]),(ua[1],ub[1])))
    T = np.matrix(np.diag((a,b)))
    M = U * T * np.transpose(U * T)
    #print M
    sigma = np.sqrt(np.diag(M))
    rho = np.sqrt(M[0,1] * M[1,0]) / np.prod(sigma)
    return sigma,rho

def test(N=32):
    X = np.zeros((N**2,N**2))
    for i in range(N):
        for j in range(N):
            r = j+i*N
            x = (i - (N-1)*0.5) / np.double(N)
            y = (j - (N-1)*0.5) / np.double(N)
            ecc = (x**2.0+y**2.0)**0.5 / 0.5**0.5 * 0.9
            #ecc = 0.0
            sigma,rho = ellipse2covariance((x,y),\
                (N / 8.0)*(1.0-ecc**2.0)**0.5,\
                ecc)
            h = binormal(shape=N,sigma=sigma,\
                #mu=(y*np.double(N),x*np.double(N)),\
                rho=rho)
            X[r,:] = np.ravel(h)
    return X
