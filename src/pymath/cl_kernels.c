#if !(defined IS_CPU)
#   error "IS_CPU not defined."
#endif
#define fmodulo(x, y) x-y*floor(x/y)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/* Simple tests

 */
__kernel void add_f32(__global const float *a, __global const float *b, __global float *c, uint sz)
{
  /*
arithmetic add for 32-bits single-precision floats.

c = a+b
   */
  uint count  = sz / get_global_size(0);
  uint idx    = (IS_CPU) ? get_global_id(0) * count : get_global_id(0);
  uint stride = (IS_CPU) ? 1 : get_global_size(0);

  for (uint n = 0; n < count; n++, idx+=stride) {
    c[idx] = a[idx]+b[idx]; // store result in global memory
  }
}

__kernel void pow_f32(__global float *a, uint sz, float b)
{
  /*
power function for 32-bits single-precision floats.
c = a^b
   */
  uint count  = sz / get_global_size(0);
  uint idx    = (IS_CPU) ? get_global_id(0) * count : get_global_id(0);
  uint stride = (IS_CPU) ? 1 : get_global_size(0);

  for (uint n = 0; n < count; n++, idx+=stride) {
    a[idx] = pow(a[idx], b);
  }
}


/* Color space related functions.

RGB stands for sRGB color space.
HSL stands for hue, staturation, lightness (not luma) color space.
L (lightness) = 0.5*max(R, G, B) + 0.5*min(R, G, B)
Y709  (luma)  = 0.2126*R + 0.7152*G + 0.0722*B (Rec. 709, HDTV)
Y2020 (luma)  = 0.2627*R + 0.6780*G + 0.0593*B (Rec. 2020, UHDTV, HDR)
HSV stands for hue, saturation, value color space.
V (value) = (R+G+B)/3.0

 */
inline float3 rgb_to_hsl(float3 rgb) {
  float3 hsl;
  float maximum = fmax(rgb.x, fmax(rgb.y, rgb.z));
  float minimum = fmin(rgb.x, fmin(rgb.y, rgb.z));
  float chroma  = maximum - minimum;
  hsl.z = .5f*(maximum+minimum);
  if (fabs(chroma) <= 1e-10) { // 2^-33 \approx 1e-10 
    hsl.x = 0.f;
    hsl.y = 0.f;
  } else {
    if (rgb.x >= fmax(rgb.y, rgb.z)) {
      hsl.x = (fmodulo((rgb.y - rgb.z)/chroma, 6.f)) / 6.f;
    } else if (rgb.y >= fmax(rgb.z, rgb.x)) {
      hsl.x = ((rgb.z - rgb.x)/chroma + 2.f) / 6.f;
    } else if (rgb.z >= fmax(rgb.x, rgb.y)) {
      hsl.x = ((rgb.x - rgb.y)/chroma + 4.f) / 6.f;
    }
    hsl.y = fmin(1.f, chroma / (1.f - fabs(2.f*hsl.z - 1.f)));
  }
  return hsl;
}

inline float3 hsl_to_rgb(float3 hsl) {
  float c = hsl.y * (1.f - fabs(2.f*hsl.z - 1.f));
  float x = c * (1.f - fabs(fmodulo(6.f*hsl.x, 2.f) - 1.f));
  float3 rgb;
  switch ((unsigned char) (6.f*hsl.x)) {
  case 0:
    rgb = (float3) (c, x, 0.f);
    break;
  case 1:
    rgb = (float3) (x, c, 0.f);
    break;
  case 2:
    rgb = (float3) (0.f, c, x);
    break;
  case 3:
    rgb = (float3) (0.f, x, c);
    break;
  case 4:
    rgb = (float3) (x, 0.f, c);
    break;
  default:
    rgb = (float3) (c, 0.f, x);
  }
  return rgb+(hsl.z-.5f*c);
}

__kernel void rgb_to_hsl_u8(
			    __global const    unsigned char *r,
			    __global const    unsigned char *g,
			    __global const    unsigned char *b,
			    __global volatile unsigned char *h,
			    __global volatile unsigned char *s,
			    __global volatile unsigned char *l,
			                               uint sz) {
  /*
color system convertion (RGB to HSL) for 8-bits unsigned integers.

   */
  uint  count  = sz / get_global_size(0);
  uint  idx    = (IS_CPU) ? get_global_id(0) * count : get_global_id(0);
  uint  stride = (IS_CPU) ? 1 : get_global_size(0);
  float3 hsl;
  for (uint n = 0; n < count; n++, idx+=stride) {
    hsl = rgb_to_hsl(((float3) (r[idx], g[idx], b[idx])) / 255.f);
    h[idx] = 255.f*hsl.x+.5f;
    s[idx] = 255.f*hsl.y+.5f;
    l[idx] = 255.f*hsl.z+.5f;
  }
}

__kernel void rgb_to_hsl_u16(
			    __global const    unsigned short *r,
			    __global const    unsigned short *g,
			    __global const    unsigned short *b,
			    __global volatile unsigned short *h,
			    __global volatile unsigned short *s,
			    __global volatile unsigned short *l,
			                                uint sz) {
  /*
color system convertion (RGB to HSL) for 16-bits unsigned integers.

   */
  uint  count  = sz / get_global_size(0);
  uint  idx    = (IS_CPU) ? get_global_id(0) * count : get_global_id(0);
  uint  stride = (IS_CPU) ? 1 : get_global_size(0);
  float3 hsl;
  for (uint n = 0; n < count; n++, idx+=stride) {
    hsl = rgb_to_hsl(((float3) (r[idx], g[idx], b[idx])) / 65535.f);
    h[idx] = 65535.f*hsl.x+.5f;
    s[idx] = 65535.f*hsl.y+.5f;
    l[idx] = 65535.f*hsl.z+.5f;
  }
}

__kernel void rgb_to_hsl_u32(
			    __global const    unsigned int *r,
			    __global const    unsigned int *g,
			    __global const    unsigned int *b,
			    __global volatile unsigned int *h,
			    __global volatile unsigned int *s,
			    __global volatile unsigned int *l,
			                              uint sz) {
  /*
color system convertion (RGB to HSL) for 32-bits unsigned integers.

   */
  uint  count  = sz / get_global_size(0);
  uint  idx    = (IS_CPU) ? get_global_id(0) * count : get_global_id(0);
  uint  stride = (IS_CPU) ? 1 : get_global_size(0);
  float3 hsl;
  for (uint n = 0; n < count; n++, idx+=stride) {
    hsl = rgb_to_hsl(((float3) (r[idx], g[idx], b[idx])) / 4294967295.f);
    h[idx] = 4294967295.f*hsl.x+.5f;
    s[idx] = 4294967295.f*hsl.y+.5f;
    l[idx] = 4294967295.f*hsl.z+.5f;
  }
}

__kernel void rgb_to_hsl_f32(
			    __global const    float *r,
			    __global const    float *g,
			    __global const    float *b,
			    __global volatile float *h,
			    __global volatile float *s,
			    __global volatile float *l,
			                       uint sz) {
  /*
color system convertion (RGB to HSL) for 32-bits unsigned integers.

   */
  uint  count  = sz / get_global_size(0);
  uint  idx    = (IS_CPU) ? get_global_id(0) * count : get_global_id(0);
  uint  stride = (IS_CPU) ? 1 : get_global_size(0);
  float3 hsl;
  for (uint n = 0; n < count; n++, idx+=stride) {
    hsl = rgb_to_hsl((float3) (r[idx], g[idx], b[idx]));
    h[idx] = hsl.x;
    s[idx] = hsl.y;
    l[idx] = hsl.z;
  }
}

__kernel void hsl_to_rgb_u8(
			    __global const    unsigned char *h,
			    __global const    unsigned char *s,
			    __global const    unsigned char *l,
			    __global volatile unsigned char *r,
			    __global volatile unsigned char *g,
			    __global volatile unsigned char *b,
			                               uint sz) {
  uint  count  = sz / get_global_size(0);
  uint  idx    = (IS_CPU) ? get_global_id(0) * count : get_global_id(0);
  uint  stride = (IS_CPU) ? 1 : get_global_size(0);
  float3 rgb;
  for (uint n = 0; n < count; n++, idx+=stride) {
    rgb = hsl_to_rgb(((float3) (h[idx], s[idx], l[idx]))/255.f);
    r[idx] = 255.f * rgb.x + .5f;
    g[idx] = 255.f * rgb.y + .5f;
    b[idx] = 255.f * rgb.z + .5f;
  }
}

__kernel void hsl_to_rgb_u16(
			    __global const    unsigned short *h,
			    __global const    unsigned short *s,
			    __global const    unsigned short *l,
			    __global volatile unsigned short *r,
			    __global volatile unsigned short *g,
			    __global volatile unsigned short *b,
			                                uint sz) {
  uint  count  = sz / get_global_size(0);
  uint  idx    = (IS_CPU) ? get_global_id(0) * count : get_global_id(0);
  uint  stride = (IS_CPU) ? 1 : get_global_size(0);
  float3 rgb;
  for (uint n = 0; n < count; n++, idx+=stride) {
    rgb = hsl_to_rgb(((float3) (h[idx], s[idx], l[idx]))/65535.f);
    r[idx] = 65535.f * rgb.x + .5f;
    g[idx] = 65535.f * rgb.y + .5f;
    b[idx] = 65535.f * rgb.z + .5f;
  }
}

__kernel void hsl_to_rgb_u32(
			    __global const    unsigned int *h,
			    __global const    unsigned int *s,
			    __global const    unsigned int *l,
			    __global volatile unsigned int *r,
			    __global volatile unsigned int *g,
			    __global volatile unsigned int *b,
			                      unsigned int sz) {
  uint  count  = sz / get_global_size(0);
  uint  idx    = (IS_CPU) ? get_global_id(0) * count : get_global_id(0);
  uint  stride = (IS_CPU) ? 1 : get_global_size(0);
  float3 rgb;
  for (uint n = 0; n < count; n++, idx+=stride) {
    rgb = hsl_to_rgb(((float3) (h[idx], s[idx], l[idx]))/4294967295.f);
    r[idx] = 4294967295.f * rgb.x + .5f;
    g[idx] = 4294967295.f * rgb.y + .5f;
    b[idx] = 4294967295.f * rgb.z + .5f;
  }
}

__kernel void hsl_to_rgb_f32(
			    __global const    float *h,
			    __global const    float *s,
			    __global const    float *l,
			    __global volatile float *r,
			    __global volatile float *g,
			    __global volatile float *b,
			                       uint sz) {
  uint  count  = sz / get_global_size(0);
  uint  idx    = (IS_CPU) ? get_global_id(0) * count : get_global_id(0);
  uint  stride = (IS_CPU) ? 1 : get_global_size(0);
  float3 rgb;
  for (uint n = 0; n < count; n++, idx+=stride) {
    rgb = hsl_to_rgb((float3) (h[idx], s[idx], l[idx]));
    r[idx] = rgb.x;
    g[idx] = rgb.y;
    b[idx] = rgb.z;
  }
}

/* image stacking functions
 */
__kernel void stack_max_u16(
				   __global const    unsigned short *r,
				   __global const    unsigned short *g,
				   __global const    unsigned short *b,
				   __global volatile unsigned short *r_res,
				   __global volatile unsigned short *g_res,
				   __global volatile unsigned short *b_res,
				                  unsigned   int sz) {
  uint count  = sz / get_global_size(0);
  uint idx    = (IS_CPU) ? get_global_id(0) * count : get_global_id(0);
  uint stride = (IS_CPU) ? 1 : get_global_size(0);
  for (uint n = 0; n < count; n++, idx+=stride) {
    if (r[idx]>r_res[idx]) r_res[idx] = r[idx];
    if (g[idx]>g_res[idx]) g_res[idx] = g[idx];
    if (b[idx]>b_res[idx]) b_res[idx] = b[idx];
  }
}

__kernel void stack_add_u16_to_f32(
				   __global const unsigned short *r,
				   __global const unsigned short *g,
				   __global const unsigned short *b,
				   __global volatile       float *r_res,
				   __global volatile       float *g_res,
				   __global volatile       float *b_res,
				                  unsigned   int sz) {
  uint count  = sz / get_global_size(0);
  uint idx    = (IS_CPU) ? get_global_id(0) * count : get_global_id(0);
  uint stride = (IS_CPU) ? 1 : get_global_size(0);
  for (uint n = 0; n < count; n++, idx+=stride) {
    r_res[idx] += (float) r[idx] / 65535.f;
    g_res[idx] += (float) g[idx] / 65535.f;
    b_res[idx] += (float) b[idx] / 65535.f;
  }
}

__kernel void stack_add_u16_to_f64(
				   __global const unsigned short *r,
				   __global const unsigned short *g,
				   __global const unsigned short *b,
				   __global volatile      double *r_res,
				   __global volatile      double *g_res,
				   __global volatile      double *b_res,
				                  unsigned   int sz) {
  uint count  = sz / get_global_size(0);
  uint idx    = (IS_CPU) ? get_global_id(0) * count : get_global_id(0);
  uint stride = (IS_CPU) ? 1 : get_global_size(0);
  for (uint n = 0; n < count; n++, idx+=stride) {
    r_res[idx] += (double) r[idx] / 65535.d;
    g_res[idx] += (double) g[idx] / 65535.d;
    b_res[idx] += (double) b[idx] / 65535.d;
  }
}
