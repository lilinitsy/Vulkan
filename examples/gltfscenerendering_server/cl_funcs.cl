#include <CL/opencl.hpp>

__kernel void rgba_to_rgb(__global const char *in, __global char *out)
{
	int in_idx = get_global_id(0) * 4;
	int out_idx = get_global_id(0) * 3;

	out[out_idx] = in[in_idx];
	out[out_idx + 1] = in[in_idx + 1];
	out[out_idx + 2] = in[in_idx + 2];
}