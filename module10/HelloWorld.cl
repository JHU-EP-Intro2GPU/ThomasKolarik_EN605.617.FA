// A kernal for adding the values in a and b together into result.
__kernel void add_kernal(__global const float *a,
						__global const float *b,
						__global float *result)
{
    int gid = get_global_id(0);

    result[gid] = a[gid] + b[gid];
}

// A kernal for subtracting the values in a and b into result.
__kernel void sub_kernal(__global const float *a,
						__global const float *b,
						__global float *result)
{
    int gid = get_global_id(0);

    result[gid] = a[gid] - b[gid];
}

// A kernal for multiplying the values in a and b into result.
__kernel void mult_kernal(__global const float *a,
						__global const float *b,
						__global float *result)
{
    int gid = get_global_id(0);

    result[gid] = a[gid] * b[gid];
}

// A kernal for "oring" the values integer values of a and b into result.
__kernel void or_kernal(__global const float *a,
						__global const float *b,
						__global float *result)
{
    int gid = get_global_id(0);

    result[gid] = static_cast<int>(a[gid]) | static_cast<int>(b[gid]);
}

// A kernal for "anding" the values integer values of a and b into result.
__kernel void and_kernal(__global const float *a,
						__global const float *b,
						__global float *result)
{
    int gid = get_global_id(0);

    result[gid] = static_cast<int>(a[gid]) & static_cast<int>(b[gid]);
}

// A kernal for "exclusing oring" the values integer values of a and b into result.
__kernel void xor_kernal(__global const float *a,
						__global const float *b,
						__global float *result)
{
    int gid = get_global_id(0);

    result[gid] = static_cast<int>(a[gid]) ^ static_cast<int>(b[gid]);
}