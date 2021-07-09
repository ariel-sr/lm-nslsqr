// Implementation of some general use kernels

#include <lmnslsqr/kernel.h>

// Modify a single value
__global__
void kernel_mod_value(double *dst, int idx, double val)
{
	dst[idx] = val;
}

// Set an array with a fixed value
__global__
void kernel_set_value(double *dst, int M, double val)
{
	// Get position
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < M)
		dst[i] =  val;
}
