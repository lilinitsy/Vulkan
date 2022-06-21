#include <chrono>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


// num_elements should include the number of channels; RGBA is 4 elements
char *allocate_device_buffer(size_t num_elements)
{
	char *buffer;
	cudaMalloc((void**) &buffer, num_elements);
	return buffer;
}

void copy_to_device_buffer(char *host, char *device, size_t num_elements)
{
	cudaMemcpy(device, host, num_elements, cudaMemcpyHostToDevice);
}

void copy_from_device_buffer(char *host, char *device, size_t num_elements)
{
	cudaMemcpy(host, device, num_elements, cudaMemcpyDeviceToHost);
}


void alpha_reduce_on_device(char *in_rgba_data, char *out_rgb_data, size_t width, size_t height)
{
	size_t num_elements_rgba = width * height * 4;
	size_t num_elements_rgb = width * height * 3;
	dim3 grid_dim = {1, 1, 1};
	dim3 block_dim = {32, 32, 1};

	char *in_rgba_data_device = allocate_device_buffer(num_elements_rgba);
	char *out_rgb_data_device = allocate_device_buffer(num_elements_rgb);

	copy_to_device_buffer(in_rgba_data, in_rgba_data_device, num_elements_rgba);

	// launch kernel
	rgba_to_rgb_kernel<<<grid_dim, block_dim>>>(in_rgba_data_device, out_rgb_data_device);

	copy_from_device_buffer(out_rgb_data, out_rgb_data_device);

	cudaDeviceSynchronize();

}