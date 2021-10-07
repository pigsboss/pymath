# coding: utf-8

"""OpenCL helper functions.
"""

## python 2 and 3 compatible.
##

import pyopencl as cl
import numpy as np
import re
from time import time
from os import path
from functools import reduce

## known integrated graphic processors
IGP_NAMES = [
    'Intel(R) UHD Graphics 630'
]

preferred_device = None

def show_device(device):
    """Show OpenCL compatible device features.
"""
    print(u"===============================================================")
    print(u"Platform name:", device.platform.name)
    print(u"Platform profile:", device.platform.profile)
    print(u"Platform vendor:", device.platform.vendor)
    print(u"Platform version:", device.platform.version)
    print(u"Device name:", device.name)
    print(u"Device type:", cl.device_type.to_string(device.type))
    print(u"Device global memory: ", device.global_mem_size//1024//1024, 'MB')
    print(u"Device max clock speed:", device.max_clock_frequency, 'MHz')
    print(u"Device compute units:", device.max_compute_units)
    print(u"Device max work group size:", device.max_work_group_size)
    print(u"Device max work item sizes:", device.max_work_item_sizes)

def list_devices():
    """List OpenCL compatible devices.
"""
    devices = []
    for platform in cl.get_platforms():
        devices += platform.get_devices()
    return devices

def list_gpus():
    """List OpenCL compatible GPUs.
"""
    devices = list_devices()
    gpus = []
    for device in devices:
        if device.type == cl.device_type.GPU:
            gpus.append(device)
    return gpus

def default_device(preferred_type=cl.device_type.GPU):
    """Give me a default OpenCL compatible device.
"""
    devices = list_devices()
    for device in devices:
        if device.type == preferred_type:
            return device
    return devices[0]

def find_compute_device(device=None):
    """Find compute device for common platforms.

If ``device`` is a pyopencl.Device, return device.
If ``device`` is a string, list all available devices and return the first device whose name matches the string.
If ``device`` is None, return a preferred device according to the following rules:
  the CPU device, if there is no available dedicated accelerator (e.g., GPU) device, otherwise select the accelerator.
  the discrete accelerator device, if available, otherwise select integrated device.
  the accelerator device with the most compute units, if multiple discrete accelerators are available.
"""
    global preferred_device
    if not isinstance(device, cl.Device):
        devs = []
        gpus = [] # discrete GPUs
        igps = [] # integrated GPUs
        for platform in cl.get_platforms():
            for d in platform.get_devices():
                if d.type == cl.device_type.CPU:
                    devs.append(d)
                else:
                    devs.append(d)
                    is_integrated = False
                    for igp in IGP_NAMES:
                        if igp.lower() in d.name.lower():
                            is_integrated = True
                    if is_integrated:
                        igps.append(d)
                    else:
                        gpus.append(d)
        if device is None:
            if len(gpus)>1:
                cus = [gpu.max_compute_units for gpu in gpus]
                device = gpus[np.argmax(cus)]
            elif len(gpus)==1:
                device = gpus[0]
            elif len(igps)>1:
                cus = [igp.max_compute_units for igp in igps]
                device = igps[np.argmax(cus)]
            elif len(igps)==1:
                device = igps[0]
            else:
                device = devs[0]
        else:
            for d in devs:
                if device.lower() in d.name.lower():
                    device = d
                    break
    assert isinstance(device, cl.Device)
    return device

def select_compute_device(device=None):
    """Select compute device.
"""
    global preferred_device
    preferred_device = find_compute_device(device)
    return preferred_device

def map(*args, flags=None, source=None, kernel_name=None, device=None, magic_number=7, profiling=False):
    """Construct a ``map`` function and execute it on a specified OpenCL device.

q_0, q_1, ..., q_n = map(p_0, p_1, ..., p_m | parameters)

Map function f(p): -> q is implemented with the runtime compiled OpenCL device kernel.
The shape of each q_i is the same as the shape of each p_j.
{p_i} are copied from the host memory to the device, then the kernel is executed on
the device, and {q_i} are copied from the device to the host memory.

Arguments:
  *args        - arguments to pass to the kernel, NumPy's ND-arrays or scalars.
                 ND-arrays always come before scalars.
                 For example, args = (p_0, p_1, ..., p_m, q_0, q_1, ..., q_n, param1, param2, ...).
                 Given a simple CL kernel z = func(x, y, gamma) = y * x ** gamma, where x and y
                 are ND-arrays and gamma is a scalar power index, the arguments could be arranged
                 as (x, y, z, gamma).

  flags        - list of mem_flags for each argument.
                 This list implies which arguments are input while the rest are output.
                 When the map function is computed in-place, e.g., x = x**2, or x = x + y,
                 use the READ_WRITE flag.

  source       - OpenCL source code for the device (string).

  kernel_name  - kernel name (string).

  device       - OpenCL compute device name (string) or cl.Device instance.

  magic_number - number of wavefronts per compute unit to hide latency magically (Default: 7).

  profiling    - profile kernel time (True) or not (False).

Returns the arguments retrieved from the device.
"""
    global preferred_device
    if device is None:
        try:
            assert isinstance(preferred_device, cl.Device)
        except:
            select_compute_device()
        device = preferred_device
    else:
        device = find_compute_device(device)
    if source is None:
        with open(path.join(path.split(path.normpath(path.abspath(path.realpath(__file__))))[0], 'map_functions.c'), 'r') as fp:
            source = fp.read()
    compute_units = device.max_compute_units
    npts = args[0].size
    if device.type == cl.device_type.GPU:
        ## GPU strategy:
        ## Many wavefronts per compute unit in order to encourage the native scheduler of each compute unit.
        ws = 64 ## work size per compute unit
        global_work_size = compute_units * magic_number * ws
        if global_work_size > npts:
            global_work_size = npts
        else:
            while (global_work_size < npts) and ((npts % global_work_size) != 0):
                global_work_size += ws
        local_work_size = ws
        is_cpu = 0
    else:
        ## CPU strategy:
        ## one thread per core.
        global_work_size = compute_units
        local_work_size = 1
        is_cpu = 1
    ctx = cl.Context([device])
    if profiling:
        queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    cl_args = []
    if flags is None:
        flags = (mf.READ_WRITE,) * len(args)
    for i in range(len(args)):
        if np.size(args[i]) > 1:
            cl_args.append(cl.Buffer(ctx, flags[i], args[i].ravel().nbytes))
        else:
            cl_args.append(args[i])
    prg  = cl.Program(ctx, source).build('-DIS_CPU={:d} -DNITEMS={:d} -DCOUNT={:d}'.format(
        is_cpu, npts, int(npts/global_work_size)))
    preferred_multiple = cl.Kernel(prg, kernel_name).get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        device
    )
    knl = getattr(prg, kernel_name)
    knl.set_args(*cl_args)
    if profiling:
        t_push, t_exec, t_pull = 0., 0., 0.
        for i in range(len(args)):
            if np.size(args[i]) > 1:
                ## cl.mem_flags.READ_ONLY  = 4 = 0b100 = 0b111 - 0b011
                ## cl.mem_flags.WRITE_ONLY = 2 = 0b010 = 0b111 - 0b101
                ## cl.mem_flags.READ_WRITE = 1 = 0b001 = 0b111 - 0b110
                ##                                                 ^^
                ##                                                 rw
                if ((7-flags[i]) & 2):
                    ev_push = cl.enqueue_copy(queue, cl_args[i], args[i])
                    ev_push.wait()
                    t_push += ev_push.profile.end - ev_push.profile.start
        ev_exec = cl.enqueue_nd_range_kernel(queue, knl, (global_work_size,), (local_work_size,))
        ev_exec.wait()
        t_exec += ev_exec.profile.end - ev_exec.profile.start
        for i in range(len(args)):
            if np.size(args[i]) > 1:
                if ((7-flags[i]) & 4):
                    ev_pull = cl.enqueue_copy(queue, args[i], cl_args[i])
                    ev_pull.wait()
                    t_pull += ev_pull.profile.end - ev_pull.profile.start
        print('{:=^50}'.format(' Profiling '))
        print('{:<32}: {:.2E} seconds'.format('Copy to device time', 1e-9*t_push))
        print('{:<32}: {:.2E} seconds'.format('OpenCL kernel execution time', 1e-9*t_exec))
        print('{:<32}: {:.2E} seconds'.format('Copy from device time', 1e-9*t_pull))
    else:
        for i in range(len(args)):
            if np.size(args[i]) > 1:
                if ((7-flags[i]) & 2):
                    ev_push = cl.enqueue_copy(queue, cl_args[i], args[i])
        ev_exec = cl.enqueue_nd_range_kernel(queue, knl, (global_work_size,), (local_work_size,))
        for i in range(len(args)):
            if np.size(args[i]) > 1:
                if ((7-flags[i]) & 4):
                    ev_pull = cl.enqueue_copy(queue, args[i], cl_args[i])
    return args

def benchmark_memory_bandwidth(device=None, max_repeats=10, max_buffer_size=256*1024**2, precision=0.01, magic_number=7):
    """Bench marking memory bandwidth related performance of specified compute device.

Arguments:
  device        a pyopencl.Device instance.
  max_repeats   maximum number of repeated runs of benchmark to average scores.
  precision     required relative precision of benchmark score.
  magic_number  number of wavefronts per compute unit to hide latency magically.

Notes:
  This function is tuned for AMD GPU especially the GCN (Graphics Core Next) microarchitecture/
  instruction set. The latest GPU series that adopts this microarchitecture is Radeon Vega.
  All recent discrete graphics card including AMD Radeon Pro Vega 20, AMD Radeon Pro Vega and
  AMD Radeon Pro Vega II that Apple Inc. equips with its pro-line products such as MacBook Pro
  15-inch, iMac Pro, Mac Pro and so on, fall in this series. Surprisingly we find a gaming card
  AMD Radeon VII (released in 2019) is not only compatible with macOS via an thunderbolt eGPU
  enclosure but especially competitive in scientific computing since it comes with 3.5TFLOPs
  double-precision computing power.

  Data transfer between the host and the compute device as well as between the global memory
  of the compute device and local registers of its compute units are relatively slow. Such
  latency could be hidden by using multiple threads. A group of threads is the minimum size of
  data to be processed in Single Instruction Many Data (SIMD) way.

  The underlying infrastructure of a compute device is its compute unit. For example there are
  20 compute unites in each AMD Radeon Pro Vega 20 GPU. Each compute unit contains 4 Texture
  Mapping Units (TMUs). Each TMU contains 16 shaders. Historically TMUs and shaders are employed
  to rotate, resize and distort bitmap images and now they are capable to perform arbitrary
  functions in GPGPU (General-purpose computing on GPU). Each TMU makes a 16-lane SIMD vector
  units (SIMD-VU), which executes 16 float or integer arithmetic operations per cycle, or 16
  4D vector arithmetic operations in 4 cycles. So each CU (compute unit) performs a single
  instruction on 4 x 16 = 64 threads simultaneously. Such a group of threads is referred to as
  a wavefront in OpenCL or a warp in CUDA context.

"""
    global preferred_device
    if device is None:
        try:
            assert isinstance(preferred_device, cl.Device)
        except:
            select_compute_device()
        device = preferred_device
    else:
        device = find_compute_device(device)
    global_mem_size = device.global_mem_size
    max_malloc_size = device.max_mem_alloc_size ## maximum global memory allocation size for a single buffer in nbytes
    compute_units = device.max_compute_units
    max_work_item_sizes = device.max_work_item_sizes ## maximum local sizes on each dimension
    max_work_group_size = device.max_work_group_size ## maximum local size
    max_work_group_dims = device.max_work_item_dimensions
    npts = int(min(max_buffer_size/4, max_malloc_size/4))
    devsrc_memcopy="""
#if !(defined IS_CPU)
#   error "IS_CPU not defined."
#endif
#if !(defined NITEMS)
#   error "NITEMS not defined."
#endif
#if !(defined COUNT)
#   error "COUNT not defined."
#endif
__kernel void memcopy(
    __global const float *a,
    __global       float *b
) {
  uint idx    = (IS_CPU) ? get_global_id(0) * COUNT : get_global_id(0);
  uint stride = (IS_CPU) ? 1 : get_global_size(0);
  for (uint n = 0; n < COUNT; n++, idx+=stride) {
    b[idx] = a[idx] * a[idx];
  }
}
"""
    print(u'{:=^80}'.format(' Benchmark Configuration '))
    print(u'{:<40}: {:<60}'.format('OpenCL Platform', device.platform.name))
    print(u'{:<40}: {:<60}'.format('Device Name', device.name))
    print(u'{:<40}: {:<60}'.format('OpenCL Version', device.version))
    print(u'{:<40}: {:d} MiB'.format('Global Memory Size', int(global_mem_size/1024**2)))
    print(u'{:<40}: {:d} MiB'.format('Maximum Allocatable Buffer Size', int(max_malloc_size/1024**2)))
    print(u'{:<40}: {:d}'.format('Compute Units', compute_units))
    print(u'{:<40}: {:d}'.format('Maximum Work Group Size', max_work_group_size))
    print(u'{:<40}: {:d}'.format('Maximum Work Group Dimensions', max_work_group_dims))
    print(u'{:<40}: {}'.format('Maximum Work Group Shape', max_work_item_sizes))
    if device.type == cl.device_type.GPU:
        ## GPU strategy:
        ## Many wavefronts per compute unit in order to encourage the native scheduler of each compute unit.
        print(u'{:<40}: {:<60}'.format('Device Type', 'GPU'))
        ws = 64 ## work size per compute unit
        global_work_size = compute_units * magic_number * ws
        if global_work_size > npts:
            global_work_size = npts
        else:
            while (global_work_size < npts) and ((npts % global_work_size) != 0):
                global_work_size += ws
        local_work_size = ws
        is_cpu = 0
    else:
        ## CPU strategy:
        ## one thread per core.
        print(u'{:<40}: {:<60}'.format('Device Type', 'CPU'))
        global_work_size = compute_units
        local_work_size = 1
        is_cpu = 1
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    mf = cl.mem_flags
    a_np = np.random.rand(npts).astype(np.float32)
    b_np = np.empty_like(a_np)
    a_cl = cl.Buffer(ctx, mf.READ_ONLY,  size=a_np.nbytes)
    b_cl = cl.Buffer(ctx, mf.WRITE_ONLY, size=b_np.nbytes)
    prg  = cl.Program(ctx, devsrc_memcopy).build('-w -DIS_CPU={:d} -DNITEMS={:d} -DCOUNT={:d}'.format(
        is_cpu, npts, int(npts/global_work_size)))
    preferred_multiple = cl.Kernel(prg, 'memcopy').get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        device
    )
    print(u'{:<40}: {:d}'.format('Data Size', int(npts)))
    print(u'{:<40}: {:d}'.format('Global Work Size', int(global_work_size)))
    print(u'{:<40}: {:d}'.format('Local Work Size', int(local_work_size)))
    print(u'{:<40}: {:d}'.format('Preferred Work Group Size Multiple', int(preferred_multiple)))
    i = 0
    p = 1.0
    t_push = []
    t_pull = []
    t_exec = []
    t_push_avg = 0.
    t_pull_avg = 0.
    t_exec_avg = 0.
    print(u'{:=^80}'.format(' Benchmark Records '))
    print(u'{:-<5}+{:-<20}+{:-<20}+{:-<20}'.format('', '', '', ''))
    print(u'{:<5}|{:<20}|{:<20}|{:<20}'.format('', ' t_push', ' t_exec', ' t_pull'))
    print(u'{:-<5}+{:-<20}+{:-<20}+{:-<20}'.format('', '', '', ''))
    while (i < max_repeats) and (p > precision):
        ev_push = cl.enqueue_copy(queue, a_cl, a_np)
        ev_exec = prg.memcopy(queue, (global_work_size,), (local_work_size,), a_cl, b_cl)
        ev_pull = cl.enqueue_copy(queue, b_np, b_cl)
        ##assert np.allclose(b_np, a_np)
        t_push.append(1e-9 * (ev_push.profile.end - ev_push.profile.start))
        t_exec.append(1e-9 * (ev_exec.profile.end - ev_exec.profile.start))
        t_pull.append(1e-9 * (ev_pull.profile.end - ev_pull.profile.start))
        p_push = np.abs(np.mean(t_push)-t_push_avg) / np.mean(t_push)
        p_exec = np.abs(np.mean(t_exec)-t_exec_avg) / np.mean(t_exec)
        p_pull = np.abs(np.mean(t_pull)-t_pull_avg) / np.mean(t_pull)
        t_push_avg = np.mean(t_push)
        t_exec_avg = np.mean(t_exec)
        t_pull_avg = np.mean(t_pull)
        p = max(p_push, p_exec, p_pull)
        print(u' {:>3d} | {:<8.2E} \u00b1 {:<7.1E} | {:<8.2E} \u00b1 {:<7.1E} | {:<8.2E} \u00b1 {:<7.1E}'.format(
            i, t_push_avg, np.std(t_push), t_exec_avg, np.std(t_exec), t_pull_avg, np.std(t_pull)))
        i += 1
    print(u'{:-<5}+{:-<20}+{:-<20}+{:-<20}'.format('', '', '', ''))
    print(u'{:=^80}'.format(' Benchmark Results '))
    r_host2dev = 8*npts / (np.double(t_pull) + np.double(t_push))
    r_global   = 8*npts / np.double(t_exec)
    print('{:<40}: {:.2E} \u00b1 {:.1E} GiB/s'.format('Host to Device Bandwidth', np.mean(r_host2dev)/1024**3, np.std(r_host2dev)/1024**3))
    print('{:<40}: {:.2E} \u00b1 {:.1E} GiB/s'.format('Device Global Bandwidth',  np.mean(r_global)/1024**3, np.std(r_global)/1024**3))
    return np.mean(r_host2dev), np.mean(r_global)

def bincount(x, weights=None, minlength=0, device=None, magic_number=7, profiling=True, verbose=True):
    """OpenCL implementaation of numpy.bincount.

Arguments:

  Refer to numpy.bincount for x, weights and minlength.
  x is ND-array of unsigned integers.

  Limited by OpenCL (atomic_inc, atomic_add and atomic_sub currently support integer operands only),
  weights should be None (unweighted) or ND-array of 32bit/64bit integers.

  device       is number, string or pyopencl.Device instance.
  magic_number is the minimum number of wavefronts per compute unit on GPU.
"""
    global preferred_device
    if device is None:
        try:
            assert isinstance(preferred_device, cl.Device)
        except:
            select_compute_device()
        device = preferred_device
    else:
        device = find_compute_device(device)
    global_mem_size = device.global_mem_size
    max_malloc_size = device.max_mem_alloc_size ## maximum global memory allocation size for a single buffer in nbytes
    compute_units = device.max_compute_units
    max_work_item_sizes = device.max_work_item_sizes ## maximum local sizes on each dimension
    max_work_group_size = device.max_work_group_size ## maximum local size
    max_work_group_dims = device.max_work_item_dimensions
    devsrc_bincount="""
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#if !(defined IS_CPU)
#   error "IS_CPU not defined."
#endif
#if !(defined NITEMS)
#   error "NITEMS not defined."
#endif
#if !(defined COUNT)
#   error "COUNT not defined."
#endif
__kernel void bincount_weighted_uint32(
             __global const unsigned int *x,
             __global const unsigned int *w,
    volatile __global                int *c
) {
  uint idx    = (IS_CPU) ? get_global_id(0) * COUNT : get_global_id(0);
  uint stride = (IS_CPU) ? 1 : get_global_size(0);
  for (uint n = 0; n < COUNT; n++, idx+=stride) {
    atomic_add(&c[x[idx]], (unsigned int) w[idx]);
  }
}

__kernel void bincount_unweighted_uint32(
             __global const unsigned int *x,
    volatile __global                int *c
) {
  uint idx    = (IS_CPU) ? get_global_id(0) * COUNT : get_global_id(0);
  uint stride = (IS_CPU) ? 1 : get_global_size(0);
  for (uint n = 0; n < COUNT; n++, idx+=stride) {
    atomic_inc(&c[x[idx]]);
  }
}

__kernel void bincount_weighted_uint64(
             __global const unsigned long *x,
             __global const unsigned long *w,
    volatile __global                long *c
) {
  unsigned long idx    = (IS_CPU) ? get_global_id(0) * COUNT : get_global_id(0);
  unsigned long stride = (IS_CPU) ? 1 : get_global_size(0);
  for (uint n = 0; n < COUNT; n++, idx+=stride) {
    atom_add(&c[x[idx]], (unsigned long) w[idx]);
  }
}

__kernel void bincount_unweighted_uint64(
             __global const unsigned long *x,
    volatile __global                long *c
) {
  unsigned long idx    = (IS_CPU) ? get_global_id(0) * COUNT : get_global_id(0);
  unsigned long stride = (IS_CPU) ? 1 : get_global_size(0);
  for (uint n = 0; n < COUNT; n++, idx+=stride) {
    atom_inc(&c[x[idx]]);
  }
}
"""
    if verbose:
        print(u'{:=^80}'.format(' OpenCL Device '))
        print(u'{:<40}: {:<60}'.format('OpenCL Platform', device.platform.name))
        print(u'{:<40}: {:<60}'.format('Device Name', device.name))
        print(u'{:<40}: {:<60}'.format('OpenCL Version', device.version))
        print(u'{:<40}: {:d} MiB'.format('Global Memory Size', int(global_mem_size/1024**2)))
        print(u'{:<40}: {:d} MiB'.format('Maximum Allocatable Buffer Size', int(max_malloc_size/1024**2)))
        print(u'{:<40}: {:d}'.format('Compute Units', compute_units))
        print(u'{:<40}: {:d}'.format('Maximum Work Group Size', max_work_group_size))
        print(u'{:<40}: {:d}'.format('Maximum Work Group Dimensions', max_work_group_dims))
        print(u'{:<40}: {}'.format('Maximum Work Group Shape', max_work_item_sizes))
    weighted  = False
    if weights is not None:
        weighted = True
    nbins = max(minlength, 1+np.max(x))
    npts  = np.size(x)
    if device.type == cl.device_type.GPU:
        ## GPU strategy:
        ## Many wavefronts per compute unit in order to encourage the native scheduler of each compute unit.
        print(u'{:<40}: {:<60}'.format('Device Type', 'GPU'))
        ws = 64 ## work size per compute unit
        global_work_size = compute_units * magic_number * ws
        if global_work_size > npts:
            global_work_size = npts
        else:
            while (global_work_size < npts) and ((npts % global_work_size) != 0):
                global_work_size += ws
        local_work_size = ws
        is_cpu = 0
    else:
        ## CPU strategy:
        ## one thread per core.
        print(u'{:<40}: {:<60}'.format('Device Type', 'CPU'))
        global_work_size = compute_units
        local_work_size = 1
        is_cpu = 1
    ctx = cl.Context([device])
    if profiling:
        queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    c_np = np.zeros((nbins,), dtype=x.dtype)
    x_cl = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=x)
    c_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=c_np)
    prg  = cl.Program(ctx, devsrc_bincount).build('-DIS_CPU={:d} -DNITEMS={:d} -DCOUNT={:d}'.format(
        is_cpu, npts, int(npts/global_work_size)))
    if verbose:
        print(u'{:=^80}'.format(' Problem Size '))
        print(u'{:<40}: {:d}'.format('Number of Data Points', int(npts)))
        print(u'{:<40}: {:d}'.format('Number of Bins', int(nbins)))
        print(u'{:<40}: {:d}'.format('Global Work Size', int(global_work_size)))
        print(u'{:<40}: {:d}'.format('Local Work Size', int(local_work_size)))
    if weighted:
        w_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weights)
        if x.dtype == np.uint32:
            ev_exec = prg.bincount_weighted_uint32(queue, (global_work_size,), (local_work_size,), x_cl, w_cl, c_cl)
        else:
            ev_exec = prg.bincount_weighted_uint64(queue, (global_work_size,), (local_work_size,), x_cl, w_cl, c_cl)
    else:
        if x.dtype == np.uint32:
            ev_exec = prg.bincount_unweighted_uint32(queue, (global_work_size,), (local_work_size,), x_cl, c_cl)
        else:
            ev_exec = prg.bincount_unweighted_uint64(queue, (global_work_size,), (local_work_size,), x_cl, c_cl)
    cl.enqueue_copy(queue, c_np, c_cl)
    if profiling:
        print(u'{:=^80}'.format(' Profiling '))
        print(u'{:<40}: {:.2E} seconds'.format('OpenCL Kernel Time', 1e-9 * (ev_exec.profile.end - ev_exec.profile.start)))
        tic = time()
        c_ref = np.bincount(x, weights=weights, minlength=minlength)
        print(u'{:<40}: {:.2E} seconds'.format('Reference Time (NumPy on CPU)', time() - tic))
        assert np.allclose(c_np, c_ref)
    return c_np

def benchmark_healpix(subroutine, nside, input_args, output_args, device=None, magic_number=7, profiling=True, verbose=True):
    """OpenCL implementaation of HEALPix subroutines.

Arguments:
    subroutine   is name of HEALPix subroutine.
        Only the following pixelwise subroutines are supported:
            pix = ang2pix_nest(theta, phi)
            theta, phi = pix2ang_ring(pix)
            pix_nest = ring2nest(pix_ring)
            pix_ring = nest2ring(pix_nest)
    nside        is the healpix nside parameter.
    input_args   is tuple of input buffers.
    output_args  is tuple of output buffers.
    device       is number, string or pyopencl.Device instance.
    magic_number is the minimum number of wavefronts per compute unit on GPU.
"""
    global preferred_device
    if device is None:
        try:
            assert isinstance(preferred_device, cl.Device)
        except:
            select_compute_device()
        device = preferred_device
    else:
        device = find_compute_device(device)
    global_mem_size = device.global_mem_size
    max_malloc_size = device.max_mem_alloc_size ## maximum global memory allocation size for a single buffer in nbytes
    compute_units = device.max_compute_units
    max_work_item_sizes = device.max_work_item_sizes ## maximum local sizes on each dimension
    max_work_group_size = device.max_work_group_size ## maximum local size
    max_work_group_dims = device.max_work_item_dimensions
    with open(path.join(path.split(path.normpath(path.abspath(path.realpath(__file__))))[0], 'healpix.c'), 'r') as fp:
        devsrc = fp.read()
    if verbose:
        print(u'{:=^80}'.format(' OpenCL Device '))
        print(u'{:<40}: {:<60}'.format('OpenCL Platform', device.platform.name))
        print(u'{:<40}: {:<60}'.format('Device Name', device.name))
        print(u'{:<40}: {:<60}'.format('OpenCL Version', device.version))
        print(u'{:<40}: {:d} MiB'.format('Global Memory Size', int(global_mem_size/1024**2)))
        print(u'{:<40}: {:d} MiB'.format('Maximum Allocatable Buffer Size', int(max_malloc_size/1024**2)))
        print(u'{:<40}: {:d}'.format('Compute Units', compute_units))
        print(u'{:<40}: {:d}'.format('Maximum Work Group Size', max_work_group_size))
        print(u'{:<40}: {:d}'.format('Maximum Work Group Dimensions', max_work_group_dims))
        print(u'{:<40}: {}'.format('Maximum Work Group Shape', max_work_item_sizes))
    npts  = np.size(input_args[0])
    if device.type == cl.device_type.GPU:
        ## GPU strategy:
        ## Many wavefronts per compute unit in order to encourage the native scheduler of each compute unit.
        print(u'{:<40}: {:<60}'.format('Device Type', 'GPU'))
        ws = 64 ## work size per compute unit
        global_work_size = compute_units * magic_number * ws
        if global_work_size > npts:
            global_work_size = npts
        else:
            while (global_work_size < npts) and ((npts % global_work_size) != 0):
                global_work_size += ws
        local_work_size = ws
        is_cpu = 0
    else:
        ## CPU strategy:
        ## one thread per core.
        print(u'{:<40}: {:<60}'.format('Device Type', 'CPU'))
        global_work_size = compute_units
        local_work_size = 1
        is_cpu = 1
    ctx = cl.Context([device])
    if profiling:
        queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    input_buffers = []
    for arg in input_args:
        input_buffers.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arg))
    output_buffers = []
    for arg in output_args:
        output_buffers.append(cl.Buffer(ctx, mf.WRITE_ONLY, size=arg.nbytes))
    opts = '-DIS_CPU={:d} -DNITEMS={:d} -DCOUNT={:d} -DNSIDE={:d}'.format(is_cpu, npts, int(npts/global_work_size), nside)
    if input_args[0].itemsize == 8:
        opts += ' -DREQUIRE_64BIT_PRECISION'
    prg  = cl.Program(ctx, devsrc).build(opts)
    if verbose:
        print(u'{:=^80}'.format(' Problem Size '))
        print(u'{:<40}: {:d}'.format('Number of Data Points', int(npts)))
        print(u'{:<40}: {:d}'.format('Global Work Size', int(global_work_size)))
        print(u'{:<40}: {:d}'.format('Local Work Size', int(local_work_size)))
    for k in prg.all_kernels():
        if k.function_name == subroutine:
            knl = k
            break
    ev_exec = knl(queue, (global_work_size,), (local_work_size,), *input_buffers, *output_buffers)
    for i in range(len(output_args)):
        cl.enqueue_copy(queue, output_args[i], output_buffers[i])
    if profiling:
        print(u'{:=^80}'.format(' Profiling '))
        print(u'{:<40}: {:.2E} seconds'.format('OpenCL Kernel Time', 1e-9 * (ev_exec.profile.end - ev_exec.profile.start)))
    return

def benchmark_device(device=None, data_points=2**23):
    """Benchmark selected device

Syntax:
benchmark_device(dev_id, data_points)
device      - compute device.
data_points - required number of data points.

"""
    global preferred_device
    if device is None:
        try:
            assert isinstance(preferred_device, cl.Device)
        except:
            select_compute_device()
        device = preferred_device
    else:
        device = find_compute_device(device)
    compute_units = device.max_compute_units
    print(" {:-^78} ".format(""))
    print("| {:<20} : {:<53} |".format("OpenCL Platform", device.platform.name))
    print("| {:<20} : {:<53} |".format("Device Name",     device.name))
    if device.type == cl.device_type.GPU:
        print("| {:<20} : {:<53} |".format("Device Type", "GPU"))
        ws = 64
        global_work_size = compute_units * 7 * ws
        while (data_points > global_work_size) and \
            ((data_points % global_work_size) != 0):
            global_work_size += ws
        if data_points < global_work_size:
            global_work_size = data_points
        local_work_size = ws
        is_cpu = 0
    else:
        print("| {:<20} : {:<53} |".format("Device Type", "CPU"))
        global_work_size = compute_units * 1
        local_work_size = 1
        is_cpu = 1
    num_groups = global_work_size / local_work_size
    print("| {:<20} : {:<53} |".format("Compute Unites",    compute_units))
    print("| {:<20} : {:<53} |".format("Global Work Size",  global_work_size))
    print("| {:<20} : {:<53} |".format("Local Work Size",   local_work_size))
    print("| {:<20} : {:<53} |".format("Number of Groups",  num_groups))
    print("| {:<20} : {:<53} |".format("Data Points",       data_points))
    print(" {:-^78} ".format(""))
    ctx   = cl.Context([device])
    mf    = cl.mem_flags
    queue = cl.CommandQueue(
        ctx, 
        properties=cl.command_queue_properties.PROFILING_ENABLE
    )
    opt_str = "-DIS_CPU=%d -DNITEMS=%d -DCOUNT=%d"%(is_cpu, data_points, int(data_points/global_work_size))
    a = np.random.rand(data_points).astype(np.float32)
    b = np.random.rand(data_points).astype(np.float32)
    c_result = np.empty_like(a)

    # Speed in normal CPU usage
    time1 = time()
    c_temp = (a+b) # adds each element in a to its corresponding element in b
    c_temp = c_temp * c_temp # element-wise multiplication
    c_result = c_temp * (a/2.0) # element-wise half a and multiply
    time2 = time()
    c_ref = c_result
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, b.nbytes)

    prg = cl.Program(ctx, """
#if !(defined IS_CPU)
#   error "IS_CPU not defined."
#endif
#if !(defined NITEMS)
#   error "NITEMS not defined."
#endif
#if !(defined COUNT)
#   error "COUNT not defined."
#endif
__kernel void sum(__global const float *a, __global const float *b, __global float *c)
{
    uint  idx    = (IS_CPU) ? get_global_id(0) * COUNT : get_global_id(0);
    uint  stride = (IS_CPU) ? 1 : get_global_size(0);
    float a_temp;
    float b_temp;
    float c_temp;

    for (uint n = 0; n < COUNT; n++, idx+=stride) {
        a_temp = a[idx]; // my a element (by global ref)
        b_temp = b[idx]; // my b element (by global ref)
        c_temp = a_temp+b_temp; // sum of my elements
        c_temp = c_temp * c_temp; // product of sums
        c_temp = c_temp * (a_temp/2.0); // times 1/2 my a
        c[idx] = atan2(c_temp,c_temp); // store result in global memory
    }
}
"""
    ).build(opt_str)

    preferred_multiple = cl.Kernel(prg, 'sum').get_work_group_info( \
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, \
        device)

    print(u"Data points:", data_points)
    print(u"Workers:", global_work_size)
    print(u"Preferred work group size multiple:", preferred_multiple)

    exec_evt = prg.sum(queue, (global_work_size,), (local_work_size,), a_buf, b_buf, dest_buf)
    exec_evt.wait()
    cl.enqueue_copy(queue, dest_buf, c_result).wait()
    elapsed = 1e-9*(exec_evt.profile.end - exec_evt.profile.start)

    print(u"Execution time of test: %g s" % elapsed)
    print(u"Acceleration: %f x CPU performance."%((time2-time1) / elapsed))
    print(u"Error: %f"%(np.std(c_ref - c_result) / np.mean((c_ref + c_result)**2.0)**0.5))

def test(device,in_vec,workers=2**9,local_size=2**8):
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx, 
            properties=cl.command_queue_properties.PROFILING_ENABLE)
    mf = cl.mem_flags
    data_points = in_vec.size
    out_vec = np.empty(workers,dtype=np.float32)
    in_vec_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=in_vec)
    out_vec_buf = cl.Buffer(ctx, mf.WRITE_ONLY, workers*4)

    time1 = time()
    sum_in = np.sum(in_vec)
    time2 = time()

    prg = cl.Program(ctx, """
        __kernel void sum(__global float* inVector,
        __global float* outVector,
        const int inVectorSize) {
            int gid = get_global_id(0);
            int gsize = get_global_size(0);
            //int wid = get_local_id(0);
            //int wsize = get_local_size(0);
            int workAmount = inVectorSize/gsize;
            int i;
            outVector[gid] = 0.0f;
            for(i=gid*workAmount;i<(gid+1)*workAmount;i++){
                outVector[gid] += inVector[i];
            }
        }
        """).build()

    preferred_multiple = cl.Kernel(prg, 'sum').get_work_group_info( \
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, \
        device)

    print(u"Data points:", data_points)
    print(u"Workers:", workers)
    print(u"Preferred work group size multiple:", preferred_multiple)

    if (local_size % preferred_multiple):
        print(u"Number of workers not a preferred multiple (%d*N)." \
                % (preferred_multiple))
        print(u"Performance may be reduced.")

    exec_evt = prg.sum(queue, (workers,), (local_size,), in_vec_buf, out_vec_buf,\
        np.uint32(data_points))
    exec_evt.wait()
    elapsed = 1e-9*(exec_evt.profile.end - exec_evt.profile.start)
    print("Acceleration: %f x CPU performance."%((time2-time1) / elapsed))
    cl.enqueue_read_buffer(queue, out_vec_buf, out_vec).wait()
    return out_vec

dot_prod_kernel_fp64="""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void dot_prod(
                __global double4 *a,    // left operant
                __global double4 *b,    // right operant
                __global double  *gsum, // group sum
                __local  double  *lsum, // local (private) sum
                  const  uint   nitems,// number of items
                  const  char   is_cpu // 0: device_is_not_cpu; 1: device_is_cpu
                 ) {
    uint   count       = (nitems / 4) / get_global_size(0);
    uint   idx         = (is_cpu) ? get_global_id(0) * count : get_global_id(0);
    uint   stride      = (is_cpu) ? 1 : get_global_size(0);
    double psum        = 0.0d;
    uint   local_idx   = get_local_id(0);

    for( uint n = 0; n < count; n++,idx+=stride )
        psum += dot(a[idx],b[idx]);
    lsum[local_idx] = psum;

    barrier( CLK_LOCAL_MEM_FENCE );
    for( int offset = get_local_size(0) / 2; offset > 0; offset >>= 1 ) {
        if( local_idx < offset )
            lsum[local_idx] += lsum[local_idx + offset];
        barrier( CLK_LOCAL_MEM_FENCE );
    }
    if( local_idx == 0 )
        gsum[ get_group_id(0) ] = lsum[0];
}
"""

dot_prod_kernel="""
__kernel void dot_prod(
                __global float4 *a,    // left operant
                __global float4 *b,    // right operant
                __global float  *gsum, // group sum
                __local  float  *lsum, // local (private) sum
                  const  uint   nitems,// number of items
                  const  char   is_cpu // 0: device_is_not_cpu; 1: device_is_cpu
                 ) {
    uint  count       = (nitems / 4) / get_global_size(0);
    uint  idx         = (is_cpu) ? get_global_id(0) * count : get_global_id(0);
    uint  stride      = (is_cpu) ? 1 : get_global_size(0);
    float psum        = (float) 0.0;
    uint  local_idx   = get_local_id(0);

    for( uint n = 0; n < count; n++,idx+=stride )
        psum += dot(a[idx],b[idx]);
    lsum[local_idx] = psum;

    barrier( CLK_LOCAL_MEM_FENCE );
    for( int offset = get_local_size(0) / 2; offset > 0; offset >>= 1 ) {
        if( local_idx < offset )
            lsum[local_idx] += lsum[local_idx + offset];
        barrier( CLK_LOCAL_MEM_FENCE );
    }
    if( local_idx == 0 )
        gsum[ get_group_id(0) ] = lsum[0];
}
"""

min_kernel="""
__kernel void minp( __global uint4 *src,
                    __global uint  *gmin,
                    __local  uint  *lmin,
                    uint           nitems,
                    char           is_cpu ) {
  uint count  = (nitems / 4) / get_global_size(0);
  uint idx    = (is_cpu) ? get_global_id(0) * count : get_global_id(0);
  uint stride = (is_cpu) ? 1 : get_global_size(0);
  uint pmin   = (uint) -1;
  uint local_idx = get_local_id(0);

    for( int n=0; n<count; n++,idx+=stride) {
        pmin = min(pmin, src[idx].x);
        pmin = min(pmin, src[idx].y);
        pmin = min(pmin, src[idx].z);
        pmin = min(pmin, src[idx].w);
    }

  lmin[local_idx] = pmin;

    barrier( CLK_LOCAL_MEM_FENCE );

  for( int offset = get_local_size(0) / 2; offset > 0; offset >>= 1 ) {
    if( local_idx < offset ) {
      uint other = lmin[local_idx+offset];
      uint mine = lmin[local_idx];
      lmin[local_idx] = (mine < other) ? mine : other;
    }
    barrier( CLK_LOCAL_MEM_FENCE );
  }
  if( local_idx == 0 )
    gmin[ get_group_id(0) ] = lmin[0];
}

__kernel void reduce( __global uint  *gmin,
                      __global uint  *result) {
    int idx = get_global_id(0);
    if(idx == 0) {
        result[0] = (uint) -1;
        for(int n = 0; n < get_global_size(0); n++) {
            if(result[0] > gmin[n])
                result[0] = gmin[n];
        }
    }
}
"""

rotate_sq_kernel="""
__kernel void qrotate(const float q0,
const float q1,
const float q2,
const float q3,
__global float* r0,
__global float* r1,
__global float* r2,
__global float* r0_out,
__global float* r1_out,
__global float* r2_out) {
    // rotate 3D vectors by a single quaternion.
    int i = get_global_id(0);
    r0_out[i] = (q0*q0 + q1*q1 - q2*q2 - q3*q3)*r0[i] + 2.0*(q1*q2 - q0*q3)*r1[i] + 2.0*(q0*q2 + q1*q3)*r2[i];
    r1_out[i] = (q0*q0 - q1*q1 + q2*q2 - q3*q3)*r1[i] + 2.0*(q0*q3 + q1*q2)*r0[i] + 2.0*(q2*q3 - q0*q1)*r2[i];
    r2_out[i] = (q0*q0 - q1*q1 - q2*q2 + q3*q3)*r2[i] + 2.0*(q0*q1 + q2*q3)*r1[i] + 2.0*(q1*q3 - q0*q2)*r0[i];
}
"""

rotate_sv_kernel="""
__kernel void qrotate(__global float* q0,
__global float* q1,
__global float* q2,
__global float* q3,
const float r0,
const float r1,
const float r2,
__global float* r0_out,
__global float* r1_out,
__global float* r2_out) {
    // rotate a single 3D vector by quaternion(s).
    int i = get_global_id(0);
    r0_out[i] = (q0[i]*q0[i] + q1[i]*q1[i] - q2[i]*q2[i] - q3[i]*q3[i])*r0 + 2.0*(q1[i]*q2[i] - q0[i]*q3[i])*r1 + 2.0*(q0[i]*q2[i] + q1[i]*q3[i])*r2;
    r1_out[i] = (q0[i]*q0[i] - q1[i]*q1[i] + q2[i]*q2[i] - q3[i]*q3[i])*r1 + 2.0*(q0[i]*q3[i] + q1[i]*q2[i])*r0 + 2.0*(q2[i]*q3[i] - q0[i]*q1[i])*r2;
    r2_out[i] = (q0[i]*q0[i] - q1[i]*q1[i] - q2[i]*q2[i] + q3[i]*q3[i])*r2 + 2.0*(q0[i]*q1[i] + q2[i]*q3[i])*r1 + 2.0*(q1[i]*q3[i] - q0[i]*q2[i])*r0;
}
"""

rotate_kernel="""
__kernel void qrotate(__global float* q0,
__global float* q1,
__global float* q2,
__global float* q3,
__global float* r0,
__global float* r1,
__global float* r2,
__global float* r0_out,
__global float* r1_out,
__global float* r2_out) {
    // rotate 3D vectors by quaternions.
    int i = get_global_id(0);
    r0_out[i] = (q0[i]*q0[i] + q1[i]*q1[i] - q2[i]*q2[i] - q3[i]*q3[i])*r0[i] + 2.0*(q1[i]*q2[i] - q0[i]*q3[i])*r1[i] + 2.0*(q0[i]*q2[i] + q1[i]*q3[i])*r2[i];
    r1_out[i] = (q0[i]*q0[i] - q1[i]*q1[i] + q2[i]*q2[i] - q3[i]*q3[i])*r1[i] + 2.0*(q0[i]*q3[i] + q1[i]*q2[i])*r0[i] + 2.0*(q2[i]*q3[i] - q0[i]*q1[i])*r2[i];
    r2_out[i] = (q0[i]*q0[i] - q1[i]*q1[i] - q2[i]*q2[i] + q3[i]*q3[i])*r2[i] + 2.0*(q0[i]*q1[i] + q2[i]*q3[i])*r1[i] + 2.0*(q1[i]*q3[i] - q0[i]*q2[i])*r0[i];
}
"""

def clmin(x, device=None, magic_number=7, numit=100, profiling=True):
    """Find minimum of input array x with map-reduce.
"""
    global preferred_device
    if device is None:
        try:
            assert isinstance(preferred_device, cl.Device)
        except:
            select_compute_device()
        device = preferred_device
    else:
        device = find_compute_device(device)
    compute_units = device.max_compute_units
    npts  = np.size(x)
    if device.type == cl.device_type.GPU:
        ## GPU strategy:
        ## Many wavefronts per compute unit in order to encourage the native scheduler of each compute unit.
        print(u'{:<40}: {:<60}'.format('Device Type', 'GPU'))
        ws = 64 ## work size per compute unit
        global_work_size = compute_units * magic_number * ws
        if global_work_size > npts:
            global_work_size = npts
        else:
            while (global_work_size < npts) and ((npts % global_work_size) != 0):
                global_work_size += ws
        local_work_size = ws
        is_cpu = 0
    else:
        ## CPU strategy:
        ## one thread per core.
        print(u'{:<40}: {:<60}'.format('Device Type', 'CPU'))
        global_work_size = compute_units
        local_work_size = 1
        is_cpu = 1
    num_groups = global_work_size // local_work_size
    ctx = cl.Context([device])
    if profiling:
        queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    prg = cl.Program(ctx, min_kernel).build()
    minp = cl.Kernel(prg, "minp")
    redc = cl.Kernel(prg, "reduce")
    sizeof_uint = np.dtype('uint32').itemsize
    src_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.uint32(x))
    dst_buf = cl.Buffer(ctx, mf.READ_WRITE, size=num_groups * sizeof_uint)
    res_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=sizeof_uint)
    minp.set_args(src_buf,dst_buf,cl.LocalMemory(local_work_size * sizeof_uint),\
        np.uint32(npts),np.uint8(is_cpu))
    redc.set_args(dst_buf,res_buf)
    t = np.zeros(numit)
    for i in range(numit):
        ev_minp=cl.enqueue_nd_range_kernel(queue,minp,(global_work_size,),(local_work_size,))
        ev_redc=cl.enqueue_nd_range_kernel(queue,redc,(num_groups,),None,None,[ev_minp])
        ev_redc.wait()
        if profiling:
            t[i] = 1e-9*(ev_minp.profile.end-ev_minp.profile.start + ev_redc.profile.end-ev_redc.profile.start)
    res = np.empty(1,dtype=np.uint32)
    cl.enqueue_copy(queue,res,res_buf)
    if profiling:
        print(u'{:=^80}'.format(' Profiling '))
        print(u'{:<40}: {:.2E} \u00b1 {:.1E} seconds'.format('OpenCL Kernel Execution Time', np.mean(t), np.std(t)))
    return res[0]

def dot_prod(a,b,device=None,numit=100):
    """Compute dot-product (a dot b) with map-reduce.
""" 
    global preferred_device
    if device is None:
        try:
            assert isinstance(preferred_device, cl.Device)
        except:
            select_compute_device()
        device = preferred_device
    else:
        device = find_compute_device(device)
    num_src_items = np.size(a)
    compute_units = device.max_compute_units
    if device.type == cl.device_type.CPU:
        global_work_size = compute_units * 1
        local_work_size = 1
        is_cpu = 1
    else:
        ws = 64
        global_work_size = compute_units * 7 * ws
        while ((num_src_items / 4) > global_work_size) and \
            (((num_src_items / 4) % global_work_size) != 0):
            global_work_size += ws
        if (num_src_items / 4) < global_work_size:
            global_work_size = num_src_items / 4
        local_work_size = ws
        is_cpu = 0
    num_groups = global_work_size / local_work_size
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    prg = cl.Program(ctx, dot_prod_kernel).build()
    dotp = cl.Kernel(prg, "dot_prod")
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.float32(a))
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.float32(b))
    c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=num_groups * 4)
    dotp.set_args(a_buf,b_buf,c_buf,cl.LocalMemory(local_work_size * 4),\
        np.uint32(num_src_items),np.uint8(is_cpu))
    for i in range(numit):
        ev_dotp=cl.enqueue_nd_range_kernel(queue,dotp,(global_work_size,),(local_work_size,))
        ev_dotp.wait()
    c = np.empty(num_groups,dtype=np.float32)
    cl.enqueue_read_buffer(queue,c_buf,c).wait()
    return np.sum(c)

def rotate(quat,vector,workers=2**8,device=None):
    """Rotate input vector(s) with given quaternion(s)
"""
    global preferred_device
    if device is None:
        try:
            assert isinstance(preferred_device, cl.Device)
        except:
            select_compute_device()
        device = preferred_device
    else:
        device = find_compute_device(device)
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
    mf = cl.mem_flags
    q_size = np.size(quat[0])
    r_size = np.size(vector[0])
    local_size=(workers,)
    if q_size == 1:
        # in-place rotate 3D vector(s) by a single quaternion
        print('rotate_sq')
        r0_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vector[0])
        r1_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vector[1])
        r2_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vector[2])
        r0_out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, r_size*4)
        r1_out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, r_size*4)
        r2_out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, r_size*4)
        prg = cl.Program(ctx, rotate_sq_kernel).build()
        global_size=(r_size,)
        exec_evt = prg.qrotate(queue, global_size, local_size, \
            np.float32(quat[0]), np.float32(quat[1]), np.float32(quat[2]), np.float32(quat[3]),\
            r0_buf, r1_buf, r2_buf, r0_out_buf, r1_out_buf, r2_out_buf)
    elif r_size == 1:
        # rotate a single 3D vector by quaternion(s)
        print('rotate_sv')
        q0_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=quat[0])
        q1_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=quat[1])
        q2_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=quat[2])
        q3_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=quat[3])
        r0_out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, q_size*4)
        r1_out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, q_size*4)
        r2_out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, q_size*4)
        prg = cl.Program(ctx, rotate_sv_kernel).build()
        global_size=(q_size,)
        exec_evt = prg.qrotate(queue, global_size, local_size, \
            q0_buf, q1_buf, q2_buf, q3_buf,\
            np.float32(vector[0]), np.float32(vector[1]), np.float32(vector[2]),\
            r0_buf, r1_buf, r2_buf, r0_out_buf, r1_out_buf, r2_out_buf)
    elif r_size == q_size:
        # in-place rotate 3D vector(s) by quaternion(s)
        print('rotate')
        q0_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=quat[0])
        q1_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=quat[1])
        q2_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=quat[2])
        q3_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=quat[3])
        r0_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vector[0])
        r1_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vector[1])
        r2_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vector[2])
        r0_out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, r_size*4)
        r1_out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, r_size*4)
        r2_out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, r_size*4)
        prg = cl.Program(ctx, rotate_kernel).build()
        global_size=(q_size,)
        exec_evt = prg.qrotate(queue, global_size, local_size, \
            q0_buf, q1_buf, q2_buf, q3_buf, r0_buf, r1_buf, r2_buf, r0_out_buf, r1_out_buf, r2_out_buf)
    else:
        raise StandardError('Can not rotate %d vectors by %d quaternions.'%(r_size,q_size))
    exec_evt.wait()
    r_out = np.empty((3,global_size[0]),dtype='float32')
    cl.enqueue_read_buffer(queue, r0_out_buf, r_out[0]).wait()
    cl.enqueue_read_buffer(queue, r1_out_buf, r_out[1]).wait()
    cl.enqueue_read_buffer(queue, r2_out_buf, r_out[2]).wait()

    preferred_multiple = cl.Kernel(prg, 'qrotate').get_work_group_info( \
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, \
        device)

    print(u"Data points: %d"%(max(q_size,r_size)))
    print(u"Workers: %d"%workers)
    print(u"Preferred work group size multiple: %d"%preferred_multiple)

    if (workers % preferred_multiple):
        print(u"Number of workers not a preferred multiple (%d*N)." \
                % (preferred_multiple))
        print(u"Performance may be reduced.")

    elapsed = 1e-9*(exec_evt.profile.end - exec_evt.profile.start)
    print(u"Execution time of test: %g s" % elapsed)
    return r_out

def nscalar(a, dtype):
    """Convert an arbitrary number to a NumPy scalar in specific data type.

a     is an arbitrary number.
dtype is numpy dtype.
"""
    if hasattr(a, '__iter__'):
        return np.dtype(dtype).type(a[0])
    else:
        return np.dtype(dtype).type(a)

class Kernel(object):
    """A packaged all-in-one OpenCL kernel, including:
 - a single-device context,
 - a command queue,
 - all I/O buffers such as NumPy ND-arrays and OpenCL buffers,
 - as well as the compiled kernel.

"""
    def __init__(self, source=None, name=None, device=None, context=None):
        """Create a Kernel instance.
source       - OpenCL source code (or path to the source file) for the device (string).
name         - Name of the kernel (function name in the source code).
device       - OpenCL compute device name (string) or cl.Device instance.
context      - OpenCL Context instance. A new context will be created if not specified.
"""
        global preferred_device
        if context is None:
            if device is None:
                try:
                    assert isinstance(preferred_device, cl.Device)
                except:
                    select_compute_device()
                self.device = preferred_device
            else:
                self.device = find_compute_device(device)
            self.context = cl.Context([self.device])
        else:
            self.context = context
            if device is None:
                self.device = self.context.devices[0]
            else:
                assert self.device in self.context.devices, 'device not included in specified context.'
        if source is None:
            with open(path.join(path.split(path.normpath(path.abspath(path.realpath(__file__))))[0], 'cl_kernels.c'), 'r') as fp:
                source = fp.read()
        if path.isfile(source):
            with open(source, 'r') as fp:
                source = fp.read()
        if self.device.type == cl.device_type.GPU:
            self.is_cpu = 0
        else:
            self.is_cpu = 1
        show_device(self.device)
        self.program = cl.Program(self.context, source).build('-DIS_CPU={:d}'.format(self.is_cpu))
        self.kernel  = getattr(self.program, name)
        ## parse source code to initiate the interface of the kernel:
        m = re.match('.*__kernel\s*void\s*{}\s*\(([^()]*)\)'.format(self.kernel.function_name), \
                     re.sub('\s', ' ', self.kernel.program.source))
        args = m.group(1).split(',')
        assert self.kernel.num_args == len(args), \
            'Parsed number of arguments ({:d}) is not consistent with the compiler ({:d}).'.format(len(args), self.kernel.num_args)
        self.args = {'index':[], 'name':[], 'keywords':[], 'is_pointer':[]}
        t = 0
        for arg in args:
            params = re.sub('^\s+', '', re.sub('\*\s+', '*', arg)).split()
            kwords = ' '.join(params[:-1])
            varnam = params[-1]
            self.args['index'].append(t)
            self.args['keywords'].append(kwords)
            if varnam.startswith('*'):
                self.args['name'].append(varnam[1:])
                self.args['is_pointer'].append(True)
            else:
                self.args['name'].append(varnam)
                self.args['is_pointer'].append(False)
            t += 1

    def link_args(self, i, knl, j):
        """Link the j-th argument of the Kernel knl to the i-th argument of this Kernel.
"""
        for k in ['dtype', 'flag', 'shape', 'size', 'value']:
            self.args[k][i] = knl.args[k][j]
        self.kernel.set_arg(i, self.args['value'][i])
        
    def set_interface(self, flags=None, dtypes=None, shapes=None, magic_number=7):
        """Set interface of the kernel by defining memory flags, data types as well as shapes of all arguments.
flags        - list of mem_flags of all arguments.
               This list implies which arguments are input while the rest are output.
               When the map function is computed in-place, e.g., x = x**2, or x = x + y,
               use the READ_WRITE flag.
dtypes       - list of data types (in NumPy dtype) of all arguments.
               if you do not want to set all arguments, leave the dtype(s) of arguments as None.
shapes       - list of shape tuples of all arguments.
magic_number - Number of wavefronts per compute unit to hide latency magically (Default: 7).
"""
        if flags is None:
            flags = [None] * self.kernel.num_args
        if dtypes is None:
            dtypes = [None] * self.kernel.num_args
        if shapes is None:
            shapes = [None] * self.kernel.num_args
        if 'flag' not in self.args:
            self.args['flag'] = [None] * self.kernel.num_args
        if 'dtype' not in self.args:
            self.args['dtype'] = [None] * self.kernel.num_args
        if 'shape' not in self.args:
            self.args['shape'] = [None] * self.kernel.num_args
        if 'value' not in self.args:
            self.args['value'] = [None] * self.kernel.num_args
        if 'size' not in self.args:
            self.args['size'] = [0] * self.kernel.num_args
        for i in range(self.kernel.num_args):
            if dtypes[i] is not None:
                self.args['dtype'][i] = np.dtype(dtypes[i])
                if self.args['is_pointer'][i]:
                    if flags[i] is None:
                        if 'const' in self.args['keywords'][i]:
                            self.args['flag'][i] = cl.mem_flags.READ_ONLY
                        else:
                            raise TypeError('Memory flag of argument {:d} (name: {}) is missing.'.format(i, self.args['name'][i]))
                    else:
                        self.args['flag'][i] = flags[i]
                    assert shapes[i] is not None, 'Shape of argument {:d} (name: {}) must be specified since it points to a buffer.'.format(i, self.args['name'][i])
                    self.args['shape'][i] = shapes[i]
                    self.args['size'][i] = reduce(lambda x,y:x*y, self.args['shape'][i])
                    self.args['value'][i] = cl.Buffer(self.context, self.args['flag'][i], size=np.dtype(self.args['dtype'][i]).itemsize * self.args['size'][i])
                    self.kernel.set_arg(i, self.args['value'][i])
                else:
                    self.args['flag'][i] = None
                    self.args['shape'][i] = ()
                    self.args['size'][i] = 1
                    self.args['value'][i] = np.dtype(self.args['dtype'][i]).type(0)
        ## set topology according to the input argument with the largest number of elements.
        compute_units = self.device.max_compute_units
        npts = max(self.args['size'])
        if self.is_cpu:
            ## CPU strategy:
            ## one thread per core.
            self.global_work_size = compute_units
            self.local_work_size = 1
        else:
            ## GPU strategy:
            ## Many wavefronts per compute unit in order to encourage the native scheduler of each compute unit.
            ws = 64 ## work size per compute unit
            self.global_work_size = compute_units * magic_number * ws
            if self.global_work_size > npts:
                self.global_work_size = npts
            else:
                while (self.global_work_size < npts) and ((npts % self.global_work_size) != 0):
                    self.global_work_size += ws
            self.local_work_size = ws

    def show_interface(self):
        """Show kernel's interface.
"""
        print('{}({})'.format(self.kernel.function_name, ', '.join(self.args['name'])))
        ast = {True:'*', False:' '}
        for i in range(self.kernel.num_args):
            print('  {:>2d} {:<30} {:1} {:8}'.format(i, self.args['keywords'][i], ast[self.args['is_pointer'][i]], self.args['name'][i]))

    def __call__(self, *args, flags=None, queue=None, profiling=False, magic_number=7):
        """Set interface, push, execute and pull.
args         - Host NumPy ND-arrays. All arguments must be provided.
flags        - list of mem_flags of all arguments.
               This list implies which arguments are input while the rest are output.
               When the map function is computed in-place, e.g., x = x**2, or x = x + y,
               use the READ_WRITE flag.
magic_number - Number of wavefronts per compute unit to hide latency magically (Default: 7).
profiling    - Profile kernel time or not (Boolean).
"""
        if queue is None:
            if profiling:
                queue = cl.CommandQueue(self.context, properties=cl.command_queue_properties.PROFILING_ENABLE)
            else:
                queue = cl.CommandQueue(self.context)
        else:
            assert queue.context == self.context, 'Specified queue is not in the same context as the kernel.'
            if profiling is None:
                profiling = bool(queue.properties & cl.command_queue_properties.PROFILING_ENABLE)
            else:
                assert profiling == bool(queue.properties & cl.command_queue_properties.PROFILING_ENABLE), 'Specified queue contradicts required property (PROFILING_ENABLE).' 
        shapes=[]
        dtypes=[]
        for arg in args:
            shapes.append(np.shape(arg))
            dtypes.append(arg.dtype)
        self.set_interface(flags=flags, dtypes=dtypes, shapes=shapes, magic_number=magic_number)
        self.push(*args, queue=queue, profiling=profiling, profiling_loops=10)
        self.execute(queue=queue, profiling=profiling, profiling_loops=10)
        self.pull(*args, queue=queue, profiling=profiling, profiling_loops=10)
        return args

    def push(self, *args, queue=None, profiling=False, profiling_loops=100):
        """Copy from the host to OpenCL buffers.
args            - Host NumPy ND-arrays. All arguments must be provided even if not all arguments are meant
                  to copied. ``None`` could be used as placeholder in such cases.
queue           - OpenCL CommandQueue instance. A new queue will be created if not specified.
profiling       - Profile kernel time or not (Boolean).
profiling_loops - Number of loops to estimate time costs.

Return: the CommandQueue instance.
"""
        if queue is None:
            if profiling:
                queue = cl.CommandQueue(self.context, properties=cl.command_queue_properties.PROFILING_ENABLE)
            else:
                queue = cl.CommandQueue(self.context)
        else:
            assert queue.context == self.context, 'Specified queue is not in the same context as the kernel.'
            if profiling is None:
                profiling = bool(queue.properties & cl.command_queue_properties.PROFILING_ENABLE)
            else:
                assert profiling == bool(queue.properties & cl.command_queue_properties.PROFILING_ENABLE), 'Specified queue contradicts required property (PROFILING_ENABLE).' 
        if profiling:
            t = np.zeros(profiling_loops)
            bs = 0
            for k in range(profiling_loops):
                for i in range(self.kernel.num_args):
                    if args[i] is not None:
                        if self.args['is_pointer'][i]:
                            if ((7-self.args['flag'][i]) & 2): ## READ_WRITE or READ_ONLY
                                assert self.args['size'][i] == args[i].size, 'Size of host NumPy array mismatches with device buffer.'
                                if np.dtype(self.args['dtype'][i]) == np.dtype(args[i].dtype):
                                    ev = cl.enqueue_copy(queue, self.args['value'][i], args[i])
                                    bs += args[i].size * args[i].dtype.itemsize
                                else:
                                    ev = cl.enqueue_copy(queue, self.args['value'][i], args[i].astype(self.args['dtype'][i]))
                                    bs += args[i].size * np.dtype(self.args['dtype'][i]).itemsize
                                ev.wait()
                                t[k] += ev.profile.end - ev.profile.start
                        else:
                            self.args['value'][i] = nscalar(args[i], self.args['dtype'][i])
                            self.kernel.set_arg(i, self.args['value'][i])
            print('{:=^60}'.format(' Profiling '))
            print('{:<32}: {:.2E} \u00b1 {:.1E} seconds'.format('Copy to device time', 1e-9*np.mean(t), 1e-9*np.std(t)))
            print('{:<32}: {:.2f} GiB/s'.format('Memory bandwidth', bs/np.mean(t)))
        else:
            for i in range(self.kernel.num_args):
                if args[i] is not None:
                    if self.args['is_pointer'][i]:
                        if ((7-self.args['flag'][i]) & 2): ## READ_WRITE or READ_ONLY
                            assert self.args['size'][i] == args[i].size, 'Size of host NumPy array mismatch with device buffer.'
                            if np.dtype(self.args['dtype'][i]) == np.dtype(args[i].dtype):
                                ev = cl.enqueue_copy(queue, self.args['value'][i], args[i])
                            else:
                                ev = cl.enqueue_copy(queue, self.args['value'][i], args[i].astype(self.args['dtype'][i]))
                            ev.wait()
                    else:
                        self.args['value'][i] = nscalar(args[i], self.args['dtype'][i])
                        self.kernel.set_arg(i, self.args['value'][i])
        return queue

    def pull(self, *args, queue=None, profiling=False, profiling_loops=100):
        """Copy OpenCL buffers to the host.
args            - Host NumPy ND-arrays. All arguments must be provided even if not all arguments are meant
                  to copied. ``None`` could be used as placeholder in such cases.
queue           - OpenCL CommandQueue instance. A new queue will be created if not specified.
profiling       - Profile kernel time or not (Boolean).
profiling_loops - Number of loops to estimate time costs.

Return: the CommandQueue instance.
"""
        if queue is None:
            if profiling:
                queue = cl.CommandQueue(self.context, properties=cl.command_queue_properties.PROFILING_ENABLE)
            else:
                queue = cl.CommandQueue(self.context)
        else:
            assert queue.context == self.context, 'Specified queue is not in the same context as the kernel.'
            if profiling is None:
                profiling = bool(queue.properties & cl.command_queue_properties.PROFILING_ENABLE)
            else:
                assert profiling == bool(queue.properties & cl.command_queue_properties.PROFILING_ENABLE), 'Specified queue contradicts required property (PROFILING_ENABLE).' 
        if profiling:
            t = np.zeros(profiling_loops)
            bs = 0
            for k in range(profiling_loops):
                for i in range(self.kernel.num_args):
                    if args[i] is not None:
                        if self.args['is_pointer'][i]:
                            if ((7-self.args['flag'][i]) & 4): ## READ_WRITE or WRITE_ONLY
                                assert self.args['size'][i] == args[i].size, 'Size of host NumPy array mismatches with device buffer.'
                                if np.dtype(self.args['dtype'][i]) == np.dtype(args[i].dtype):
                                    ev = cl.enqueue_copy(queue, args[i], self.args['value'][i])
                                    bs += args[i].size * args[i].dtype.itemsize
                                    ev.wait()
                                else:
                                    tmp = np.empty(self.args['shape'][i], dtype=self.args['dtype'][i])
                                    ev = cl.enqueue_copy(queue, tmp, self.args['value'][i])
                                    bs += args[i].size * np.dtype(self.args['dtype'][i]).itemsize
                                    ev.wait()
                                    args[i][:] = tmp[:]
                                t[k] += ev.profile.end - ev.profile.start
            print('{:=^60}'.format(' Profiling '))
            print('{:<32}: {:.2E} \u00b1 {:.1E} seconds'.format('Copy to host time', 1e-9*np.mean(t), 1e-9*np.std(t)))
            print('{:<32}: {:.2f} GiB/s'.format('Memory bandwidth', bs/np.mean(t)))
        else:
            for i in range(self.kernel.num_args):
                if args[i] is not None:
                    if self.args['is_pointer'][i]:
                        if ((7-self.args['flag'][i]) & 4): ## READ_WRITE or WRITE_ONLY
                            assert self.args['size'][i] == args[i].size, 'Size of host NumPy array mismatches with device buffer.'
                            if np.dtype(self.args['dtype'][i]) == np.dtype(args[i].dtype):
                                ev = cl.enqueue_copy(queue, args[i], self.args['value'][i])
                                ev.wait()
                            else:
                                tmp = np.empty(self.args['shape'][i], dtype=self.args['dtype'][i])
                                ev = cl.enqueue_copy(queue, tmp, self.args['value'][i])
                                ev.wait()
                                args[i][:] = tmp[:]
        return queue

    def execute(self, queue=None, profiling=False, profiling_loops=100):
        """Execute the configured kernel.
queue           - OpenCL CommandQueue instance. A new queue will be created if not specified.
profiling       - Profile kernel time or not (Boolean).
profiling_loops - Number of loops to estimate time costs.

Return: the CommandQueue instance.
"""
        if queue is None:
            if profiling:
                queue = cl.CommandQueue(self.context, properties=cl.command_queue_properties.PROFILING_ENABLE)
            else:
                queue = cl.CommandQueue(self.context)
        else:
            assert queue.context == self.context, 'Specified queue is not in the same context as the kernel.'
            if profiling is None:
                profiling = bool(queue.properties & cl.command_queue_properties.PROFILING_ENABLE)
            else:
                assert profiling == bool(queue.properties & cl.command_queue_properties.PROFILING_ENABLE), 'Specified queue contradicts required property (PROFILING_ENABLE).' 
        if profiling:
            t = np.zeros(profiling_loops)
            for k in range(profiling_loops):
                ev = cl.enqueue_nd_range_kernel(queue, self.kernel, (self.global_work_size,), (self.local_work_size,))
                ev.wait()
                t[k] += ev.profile.end - ev.profile.start
            print('{:=^60}'.format(' Profiling '))
            print('{:<32}: {:.2E} \u00b1 {:.1E} seconds'.format('OpenCL kernel execution time', 1e-9*np.mean(t), 1e-9*np.std(t)))
        else:
            ev = cl.enqueue_nd_range_kernel(queue, self.kernel, (self.global_work_size,), (self.local_work_size,))
            ev.wait()
        return queue
