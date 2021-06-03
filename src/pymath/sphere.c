#if !(defined IS_CPU)
#   error "IS_CPU not defined."
#endif
#define fmodulo(x, y) x-y*floor(x/y)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define DPI     3.1415926535897932385
#define DHALFPI 1.5707963267948966192
#define DTWOPI  6.2831853071795864769
#define PI      3.1415926535897932385f
#define HALFPI  1.5707963267948966192f
#define TWOPI   6.2831853071795864769f

/*
Construct quaternion from Tait-Bryan angles phi, theta and psi.
*/
inline float4 from_angles_f32(float phi, float theta) {
  float hphi = 0.5f * phi;
  float hthe = 0.5f * theta;
  float coshphi = cos(hphi);
  float coshthe = cos(hthe);
  float sinhphi = sin(hphi);
  float sinhthe = sin(hthe);
  return (float4)
    (
      coshphi*coshthe,
      sinhphi*sinhthe,
     -coshphi*sinhthe,
      sinhphi*coshthe
     );
}
inline double4 from_angles_f64(double phi, double theta) {
  double hphi = 0.5 * phi;
  double hthe = 0.5 * theta;
  double coshphi = cos(hphi);
  double coshthe = cos(hthe);
  double sinhphi = sin(hphi);
  double sinhthe = sin(hthe);
  return (double4) (coshphi*coshthe, sinhphi*sinhthe, -coshphi*sinhthe, sinhphi*coshthe);
}

/*
Rotate input vector with given quaternion.
*/
inline float3 rotate_f32(float4 q, float3 r) {
  return (float3)
    (
     (q.x*q.x + q.y*q.y - q.z*q.z - q.w*q.w)*r.x + 2.0f*(q.y*q.z - q.x*q.w)*r.y + 2.0f*(q.x*q.z + q.y*q.w)*r.z,
     (q.x*q.x - q.y*q.y + q.z*q.z - q.w*q.w)*r.y + 2.0f*(q.z*q.w - q.x*q.y)*r.z + 2.0f*(q.x*q.w + q.y*q.z)*r.x,
     (q.x*q.x - q.y*q.y - q.z*q.z + q.w*q.w)*r.z + 2.0f*(q.y*q.w - q.x*q.z)*r.x + 2.0f*(q.x*q.y + q.z*q.w)*r.y
     );
}
inline double3 rotate_f64(double4 q, double3 r) {
  return (double3)
    (
     (q.x*q.x + q.y*q.y - q.z*q.z - q.w*q.w)*r.x + 2.0*(q.y*q.z - q.x*q.w)*r.y + 2.0*(q.x*q.z + q.y*q.w)*r.z,
     (q.x*q.x - q.y*q.y + q.z*q.z - q.w*q.w)*r.y + 2.0*(q.z*q.w - q.x*q.y)*r.z + 2.0*(q.x*q.w + q.y*q.z)*r.x,
     (q.x*q.x - q.y*q.y - q.z*q.z + q.w*q.w)*r.z + 2.0*(q.y*q.w - q.x*q.z)*r.x + 2.0*(q.x*q.y + q.z*q.w)*r.y
     );
}

/*
Convert cartesian coordinates (x,y,z) to spherical coordinates (phi, theta, rho).
*/
inline float3 xyz2ptr_f32(float3 xyz) {
  float rho   = sqrt(xyz.x*xyz.x+xyz.y*xyz.y+xyz.z*xyz.z);
  float phi   = atan2(xyz.y, xyz.x);
  float theta = atan2(xyz.z, sqrt(xyz.x*xyz.x+xyz.y*xyz.y));
  return (float3) (phi, theta, rho);
}
inline double3 xyz2ptr_f64(double3 xyz) {
  double rho   = sqrt(xyz.x*xyz.x+xyz.y*xyz.y+xyz.z*xyz.z);
  double phi   = atan2(xyz.y, xyz.x);
  double theta = atan2(xyz.z, sqrt(xyz.x*xyz.x+xyz.y*xyz.y));
  return (double3) (phi, theta, rho);
}
inline float3 ptr2xyz_f32(float3 ptr) {
  float xy = ptr.z * cos(ptr.y);
  float x  = xy * cos(ptr.x);
  float y  = xy * sin(ptr.x);
  float z  = ptr.z * sin(ptr.y);
  return (float3) (x,y,z);
}
inline double3 ptr2xyz_f64(double3 ptr) {
  double xy = ptr.z * cos(ptr.y);
  double x  = xy * cos(ptr.x);
  double y  = xy * sin(ptr.x);
  double z  = ptr.z * sin(ptr.y);
  return (double3) (x,y,z);
}

/*
Geodesic angle and distance from the point a to point b on unit sphere.

The geodesic angle is the angle from the meridian passing point a to the
geodesic from a to b.
In our convention a meridian goes from the south pole to the north pole.
Both a and b are point(s) on the unit sphere.
a or b can be given in the following ways:
  1, a is a point while b is another point.
  2, one of them is a point while the other is an array of points.
  3, both of them are arrays of points, but in this way the two arrays
  must have the same numbers of points.
Points on the spherical surface can be given in either their Cartesian
coordinates as (x,y,z).T or in polar angles as (phi,theta).T
*/
__kernel void geodesic_f32(
  __global const    float *phi_a,
  __global const    float *theta_a,
  __global const    float *phi_b,
  __global const    float *theta_b,
  __global volatile float *p,
  __global volatile float *d,
                     uint sz
) {
  uint count  = sz / get_global_size(0);
  uint idx    = (IS_CPU) ? get_global_id(0) * count : get_global_id(0);
  uint stride = (IS_CPU) ? 1 : get_global_size(0);
  for (uint n = 0; n < count; n++, idx+=stride) {
    float4 q = from_angles_f32(phi_a[idx]-PI, HALFPI-theta_a[idx]);
    float3 r = ptr2xyz_f32((float3) (phi_b[idx], theta_b[idx], 1.0f));
    r = rotate_f32((float4)(q.x,-q.y,-q.z,-q.w), r);
    r = xyz2ptr_f32(r);
    p[idx] = r.x;
    d[idx] = HALFPI-r.y;
  }
  idx = count*get_global_size(0) + get_global_id(0);
  if (idx < sz) {
    float4 q = from_angles_f32(phi_a[idx]-PI, HALFPI-theta_a[idx]);
    float3 r = ptr2xyz_f32((float3) (phi_b[idx], theta_b[idx], 1.0f));
    r = rotate_f32((float4)(q.x,-q.y,-q.z,-q.w), r);
    r = xyz2ptr_f32(r);
    p[idx] = r.x;
    d[idx] = HALFPI-r.y;
  }
}
__kernel void geodesic_f64(
  __global const    double *phi_a,
  __global const    double *theta_a,
  __global const    double *phi_b,
  __global const    double *theta_b,
  __global volatile double *p,
  __global volatile double *d,
                      uint sz
) {
  uint count  = sz / get_global_size(0);
  uint idx    = (IS_CPU) ? get_global_id(0) * count : get_global_id(0);
  uint stride = (IS_CPU) ? 1 : get_global_size(0);
  for (uint n = 0; n < count; n++, idx+=stride) {
    double4 q = from_angles_f64(phi_a[idx]-DPI, DHALFPI-theta_a[idx]);
    double3 r = ptr2xyz_f64((double3) (phi_b[idx], theta_b[idx], 1.0));
    r = rotate_f64((double4)(q.x,-q.y,-q.z,-q.w), r);
    r = xyz2ptr_f64(r);
    p[idx] = r.x;
    d[idx] = DHALFPI-r.y;
  }
  idx = count*get_global_size(0) + get_global_id(0);
  if (idx < sz) {
    double4 q = from_angles_f64(phi_a[idx]-DPI, DHALFPI-theta_a[idx]);
    double3 r = ptr2xyz_f64((double3) (phi_b[idx], theta_b[idx], 1.0));
    r = rotate_f64((double4)(q.x,-q.y,-q.z,-q.w), r);
    r = xyz2ptr_f64(r);
    p[idx] = r.x;
    d[idx] = DHALFPI-r.y;
  }
}
__kernel void geodesic_bcast_f32(
                    float phi_a,
                    float theta_a,
  __global const    float *phi_b,
  __global const    float *theta_b,
  __global volatile float *p,
  __global volatile float *d,
                     uint sz
) {
  uint count  = sz / get_global_size(0);
  uint idx    = (IS_CPU) ? get_global_id(0) * count : get_global_id(0);
  uint stride = (IS_CPU) ? 1 : get_global_size(0);
  float4 q = from_angles_f32(phi_a-PI, HALFPI-theta_a);
  for (uint n = 0; n < count; n++, idx+=stride) {
    float3 r = ptr2xyz_f32((float3) (phi_b[idx], theta_b[idx], 1.0f));
    r = rotate_f32((float4)(q.x,-q.y,-q.z,-q.w), r);
    r = xyz2ptr_f32(r);
    p[idx] = r.x;
    d[idx] = HALFPI-r.y;
  }
  idx = count*get_global_size(0) + get_global_id(0);
  if (idx<sz) {
    float3 r = ptr2xyz_f32((float3) (phi_b[idx], theta_b[idx], 1.0f));
    r = rotate_f32((float4)(q.x,-q.y,-q.z,-q.w), r);
    r = xyz2ptr_f32(r);
    p[idx] = r.x;
    d[idx] = HALFPI-r.y;
  }
}
__kernel void geodesic_bcast_f64(
                    double phi_a,
                    double theta_a,
  __global const    double *phi_b,
  __global const    double *theta_b,
  __global volatile double *p,
  __global volatile double *d,
                      uint sz
) {
  uint count  = sz / get_global_size(0);
  uint idx    = (IS_CPU) ? get_global_id(0) * count : get_global_id(0);
  uint stride = (IS_CPU) ? 1 : get_global_size(0);
  double4 q = from_angles_f64(phi_a-DPI, DHALFPI-theta_a);
  for (uint n = 0; n < count; n++, idx+=stride) {
    double3 r = ptr2xyz_f64((double3) (phi_b[idx], theta_b[idx], 1.0));
    r = rotate_f64((double4)(q.x,-q.y,-q.z,-q.w), r);
    r = xyz2ptr_f64(r);
    p[idx] = r.x;
    d[idx] = DHALFPI-r.y;
  }
  idx = count*get_global_size(0) + get_global_id(0);
  if (idx < sz) {
    double3 r = ptr2xyz_f64((double3) (phi_b[idx], theta_b[idx], 1.0));
    r = rotate_f64((double4)(q.x,-q.y,-q.z,-q.w), r);
    r = xyz2ptr_f64(r);
    p[idx] = r.x;
    d[idx] = DHALFPI-r.y;
  }
}


