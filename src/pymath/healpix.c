/* --------------------------------------------------------------------------------
 * This file is part of PyMath, a set of Python Math modules for self education and
 * test. It should by no means be imported to any productive project without careful
 * validation.
 *
 * For more information about HEALPix see http://healpix.sourceforge.net
 *
 * Update: 2019/10/05
 * ------------------------------------------------------------------------------ */
#if !(defined IS_CPU)
#error "IS_CPU not defined."
#endif
#if !(defined NITEMS)
#error "NITEMS not defined."
#endif
#if !(defined COUNT)
#error "COUNT not defined."
#endif
#if !(defined NSIDE)
#error "NSIDE not defined."
#endif
#if defined(REQUIRE_64BIT_PRECISION)
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#define TWOTHIRD   2.0/3.0
#define TWOPI      6.283185307179586476925286766559005768394
#define INV_HALFPI 0.6366197723675813430755350534900574
#define PI         3.141592653589793238462643383279502884197
#define HALFPI     1.570796326794896619231321691639751442099

/*! Returns the reminder of the division x/y.
    By pigsboss@github

    OpenCL built-in math function fmod is defined as:
    fmod(x, y) = x - y * trunc(x/y),
    where trunc is round to zero. In this way the reminder
    could be positive or negative. For example, fmod(-1, 3) == -1.
    To implement the always-positive mod function, as -1 % 3 == 2,
    we can round x/y down to negative infinity instead of zero. */

inline double fmodulo (double x, double y) {
    return x - y * floor(x/y);
}

inline long ctab (int m) {
    return  (m&0x01)
        |  ((m&0x02) << 7)
        |  ((m&0x04) >> 1)
        |  ((m&0x08) << 6)
        |  ((m&0x10) >> 2)
        |  ((m&0x20) << 5)
        |  ((m&0x40) >> 3)
        |  ((m&0x80) << 4);
}

inline long utab (int m) {
    return  (m&0x01)
        |  ((m&0x02) << 1)
        |  ((m&0x04) << 2)
        |  ((m&0x08) << 3)
        |  ((m&0x10) << 4)
        |  ((m&0x20) << 5)
        |  ((m&0x40) << 6)
        |  ((m&0x80) << 7);
}

inline long spread_bits (long v) {
    return  (utab( v     &0xff)) | ((utab((v>> 8)&0xff))<<16) | ((utab((v>>16)&0xff))<<32) | ((utab((v>>24)&0xff))<<48);
}

inline long xyf2nest (long ix, long iy, long face_num) {
    return (face_num*NSIDE*NSIDE) + spread_bits(ix) + (spread_bits(iy)<<1);
}

inline long ang2pix_nest_z_phi (double z, double phi) {
    double za = fabs(z);
    double tt = fmodulo(phi, TWOPI) * INV_HALFPI; /* in [0,4) */
    long face_num, ix, iy;
    if (za<=TWOTHIRD) {      /* Equatorial region */
        double temp1 = NSIDE*(0.5+tt);
        double temp2 = NSIDE*(z*0.75);
        long jp = (long)(temp1-temp2); /* index of  ascending edge line */
        long jm = (long)(temp1+temp2); /* index of descending edge line */
        long ifp = jp/NSIDE;  /* in {0,4} */
        long ifm = jm/NSIDE;
        face_num = (ifp==ifm) ? (ifp|4) : ((ifp<ifm) ? ifp : (ifm+8));
        ix = jm & (NSIDE-1);
        iy = NSIDE - (jp & (NSIDE-1)) - 1;
	}
    else {                   /* polar region, za > 2/3 */
        int ntt = (int) tt;
        long jp, jm;
        double tp, tmp;
        if (ntt>=4) ntt=3;
        tp = tt-ntt;
        tmp = NSIDE*sqrt(3*(1-za));
        jp = (long)(tp*tmp); /* increasing edge line index */
        jm = (long)((1.0-tp)*tmp); /* decreasing edge line index */
        if (jp>=NSIDE) jp = NSIDE-1; /* for points too close to the boundary */
        if (jm>=NSIDE) jm = NSIDE-1;
        if (z >= 0) {
            face_num = ntt;  /* in {0,3} */
            ix = NSIDE - jm - 1;
            iy = NSIDE - jp - 1;
	    }
        else {
            face_num = ntt + 8; /* in {8,11} */
            ix =  jp;
            iy =  jm;
	    }
	}
    return xyf2nest(ix,iy,face_num);
}

__kernel void ang2pix_nest(
			   __global const    double        *theta,
			   __global const    double        *phi,
			   __global volatile unsigned long *ipix) {
  unsigned long idx    = (IS_CPU) ? get_global_id(0) * COUNT : get_global_id(0);
  unsigned long stride = (IS_CPU) ? 1 : get_global_size(0);
  for (uint n = 0; n < COUNT; n++, idx+=stride) {
    ipix[idx] = (unsigned long) ang2pix_nest_z_phi(cos(theta[idx]), phi[idx]);
  }
}

__kernel void multiply(
			   __global const    long *a,
			   __global const    long *b,
			   __global          long *c) {
  unsigned long idx    = (IS_CPU) ? get_global_id(0) * COUNT : get_global_id(0);
  unsigned long stride = (IS_CPU) ? 1 : get_global_size(0);
  for (uint n = 0; n < COUNT; n++, idx+=stride) {
    c[idx] = a[idx] * b[idx];
  }
}

__kernel void sum(
			   __global const    long *a,
			   __global const    long *b,
			   __global          long *c) {
  unsigned long idx    = (IS_CPU) ? get_global_id(0) * COUNT : get_global_id(0);
  unsigned long stride = (IS_CPU) ? 1 : get_global_size(0);
  for (uint n = 0; n < COUNT; n++, idx+=stride) {
    c[idx] = a[idx] + b[idx];
  }
}

__kernel void _imax(
			   __global const    long *a,
			   __global const    long *b,
			   __global          long *c) {
  unsigned long idx    = (IS_CPU) ? get_global_id(0) * COUNT : get_global_id(0);
  unsigned long stride = (IS_CPU) ? 1 : get_global_size(0);
  for (uint n = 0; n < COUNT; n++, idx+=stride) {
    c[idx] = (a[idx]>b[idx]) ? a[idx] : b[idx];
  }
}

__kernel void _fmax(
			   __global const    long *a,
			   __global const    long *b,
			   __global          long *c) {
  unsigned long idx    = (IS_CPU) ? get_global_id(0) * COUNT : get_global_id(0);
  unsigned long stride = (IS_CPU) ? 1 : get_global_size(0);
  for (uint n = 0; n < COUNT; n++, idx+=stride) {
    c[idx] = (long) fmax((double)a[idx], (double)b[idx]);
  }
}

__kernel void _max(
			   __global const    long *a,
			   __global const    long *b,
			   __global          long *c) {
  unsigned long idx    = (IS_CPU) ? get_global_id(0) * COUNT : get_global_id(0);
  unsigned long stride = (IS_CPU) ? 1 : get_global_size(0);
  for (uint n = 0; n < COUNT; n++, idx+=stride) {
    c[idx] = max(a[idx], b[idx]);
  }
}


#else
#define TWOTHIRD   2.0f/3.0f
#define TWOPI      6.283185307179586476925286766559005768394f
#define INV_HALFPI 0.6366197723675813430755350534900574f
#define PI         3.141592653589793238462643383279502884197f
#define HALFPI     1.570796326794896619231321691639751442099f

inline float fmodulo (float x, float y) {
    return x - y * floor(x/y);
}

inline int ctab (int m) {
    return  (m&0x01)
        |  ((m&0x02) << 7)
        |  ((m&0x04) >> 1)
        |  ((m&0x08) << 6)
        |  ((m&0x10) >> 2)
        |  ((m&0x20) << 5)
        |  ((m&0x40) >> 3)
        |  ((m&0x80) << 4);
}

inline int utab (int m) {
    return  (m&0x01)
        |  ((m&0x02) << 1)
        |  ((m&0x04) << 2)
        |  ((m&0x08) << 3)
        |  ((m&0x10) << 4)
        |  ((m&0x20) << 5)
        |  ((m&0x40) << 6)
        |  ((m&0x80) << 7);
}

inline int spread_bits (int v) {
    return  (utab( v     &0xff)) | ((utab((v>> 8)&0xff))<<16) | ((utab((v>>16)&0xff))<<32) | ((utab((v>>24)&0xff))<<48);
}

inline int xyf2nest (int ix, int iy, int face_num) {
    return (face_num*NSIDE*NSIDE) + spread_bits(ix) + (spread_bits(iy)<<1);
}

inline int ang2pix_nest_z_phi (float z, float phi) {
    float za = fabs(z);
    float tt = fmodulo(phi, TWOPI) * INV_HALFPI; /* in [0,4) */
    int face_num, ix, iy;
    if (za<=TWOTHIRD) { /* Equatorial region */
        float temp1 = NSIDE*(0.5+tt);
        float temp2 = NSIDE*(z*0.75);
        int jp = (int)(temp1-temp2); /* index of  ascending edge line */
        int jm = (int)(temp1+temp2); /* index of descending edge line */
        int ifp = jp/NSIDE;  /* in {0,4} */
        int ifm = jm/NSIDE;
        face_num = (ifp==ifm) ? (ifp|4) : ((ifp<ifm) ? ifp : (ifm+8));
        ix = jm & (NSIDE-1);
        iy = NSIDE - (jp & (NSIDE-1)) - 1;
    }
    else { /* polar region, za > 2/3 */
        int ntt = (int)tt, jp, jm;
        float tp, tmp;
        if (ntt>=4) ntt=3;
        tp = tt-ntt;
        tmp = NSIDE*sqrt(3*(1-za));
        jp = (int)(tp*tmp); /* increasing edge line index */
        jm = (int)((1.0-tp)*tmp); /* decreasing edge line index */
        if (jp>=NSIDE) jp = NSIDE-1; /* for points too close to the boundary */
        if (jm>=NSIDE) jm = NSIDE-1;
        if (z >= 0) {
            face_num = ntt;  /* in {0,3} */
            ix = NSIDE - jm - 1;
            iy = NSIDE - jp - 1;
	}
        else {
            face_num = ntt + 8; /* in {8,11} */
            ix =  jp;
            iy =  jm;
	}
    }
    return xyf2nest(ix,iy,face_num);
}

__kernel void ang2pix_nest(
    __global const float *theta,
    __global const float *phi,
    __global         int *ipix) {
    unsigned int idx    = (IS_CPU) ? get_global_id(0) * COUNT : get_global_id(0);
    unsigned int stride = (IS_CPU) ? 1 : get_global_size(0);
  for (uint n = 0; n < COUNT; n++, idx+=stride) {
      ipix[idx] = ang2pix_nest_z_phi(cos(theta[idx]), phi[idx]);
  }
}
#endif
