#if !(defined IS_CPU)
#   error "IS_CPU not defined."
#endif
#if !(defined NITEMS)
#   error "NITEMS not defined."
#endif

__kernel void update_pc_loading_local_fp32(
    __global float4 * Rcv,      // Residual matrix column vector,      input
    __global float4 * Tcv,      // Score matrix column vector,         input
    __global float  * group_sum,
    __local  float  * local_sum
) {
    uint   count    = (NITEMS/4) / get_global_size(0);
    uint   idx      = (IS_CPU) ? get_global_id(0) * count : get_global_id(0);
    uint   stride   = (IS_CPU) ? 1 : get_global_size(0);
    uint   local_id = get_local_id(0);
    float  psum     = 0.0f;
    for( uint n = 0; n < count; n++,idx+=stride )
        psum += dot(Rcv[idx],Tcv[idx]);
    local_sum[local_id] = psum;
    barrier( CLK_LOCAL_MEM_FENCE );
    for( uint offset = get_local_size(0) / 2; offset > 0; offset >>= 1 ) {
        if( local_id < offset )
            local_sum[local_id] += local_sum[local_id + offset];
        barrier( CLK_LOCAL_MEM_FENCE );
    }
    if( local_id == 0 )
        group_sum[get_group_id(0)] = local_sum[0];
}

__kernel void update_pc_loading_global_fp32(
    __global float * group_sum,
    __global float * Pcv,
              uint   k
) {
    uint  num_groups = get_num_groups(0);
    float psum = 0.0f;
    if ( get_global_id(0) == 0 ) {
        for( uint n = 0; n < num_groups; n++ )
            psum += group_sum[n];
        Pcv[k] = psum;
    }
}

__kernel void update_pc_score_fp32(
    __global float4 * Rcv,      // Residual matrix column vector,      input
    __global float4 * Tcv_next, // Updated score matrix column vector, output
    __global float  * Pcv,      // Loading matrix column vector,       output
              uint    k
) {
    uint   count       = (NITEMS/4) / get_global_size(0);
    uint   idx         = (IS_CPU) ? get_global_id(0) * count : get_global_id(0);
    uint   stride      = (IS_CPU) ? 1 : get_global_size(0);
    //uint   local_id    = get_local_id(0);
    float  psum        = Pcv[k];
    for( uint n = 0; n < count; n++,idx+=stride )
        Tcv_next[idx]  = fma(Rcv[idx], psum, Tcv_next[idx]);
}

__kernel void dist_local_fp32(
    __global float4 * a,
    __global float4 * b,
    __global float  * gd2,
    __local  float  * ld2
) {
    uint   count       = (NITEMS/4) / get_global_size(0);
    uint   idx         = (IS_CPU) ? get_global_id(0) * count : get_global_id(0);
    uint   stride      = (IS_CPU) ? 1 : get_global_size(0);
    uint   local_id    = get_local_id(0);
    float  pd2      = 0.0f;
    for( uint n = 0; n < count; n++,idx+=stride )
        pd2 += dot(a[idx]-b[idx], a[idx]-b[idx]);
    ld2[local_id] = pd2;
    barrier( CLK_LOCAL_MEM_FENCE );
    for( uint offset = get_local_size(0) / 2; offset > 0; offset >>= 1 ) {
        if( local_id < offset )
            ld2[local_id] += ld2[local_id + offset];
        barrier( CLK_LOCAL_MEM_FENCE );
    }
    if( local_id == 0 )
        gd2[get_group_id(0)] = ld2[0];
}

__kernel void dist_global_fp32(
    __global float * gd2,
    __global float * d
) {
    uint  num_groups = get_num_groups(0);
    float pd2 = 0.0f;
    if ( get_global_id(0) == 0 ) {
        for( uint n = 0; n < num_groups; n++ )
            pd2 += gd2[n];
        d[0] = sqrt(pd2);
    }
}
