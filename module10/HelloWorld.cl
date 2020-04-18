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

// A kernal for dividing the values in a and b into result.
__kernel void div_kernal(__global const float *a,
						__global const float *b,
						__global float *result)
{
    int gid = get_global_id(0);

    if (b[gid] != 0.0)
    {
        result[gid] = a[gid] / b[gid];
    }
    else
    {
        result[gid] = 0.0;
    }
}

// A kernal for doing the modulus of the values in a and b into result.
__kernel void mod_kernal(__global const float *a,
						__global const float *b,
						__global float *result)
{
    int gid = get_global_id(0);

    if (b[gid] != 0.0)
    {
        result[gid] = a[gid] % b[gid];
    }
    else
    {
        result[gid] = 0.0;
    }
}

// A kernal for taking the average value of a and b and putting into result
__kernel void avg_kernal(__global const float *a,
						__global const float *b,
						__global float *result)
{
    int gid = get_global_id(0);

    result[gid] = (a[gid] + b[id]) / 2.0;
}