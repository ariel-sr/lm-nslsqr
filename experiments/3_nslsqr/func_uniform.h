#ifndef _FUNC_UNIFORM_H
#define _FUNC_UNIFORM_H 

#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <lmnslsqr/kernel.h>
#include <curand_kernel.h>
#include <curand.h>

////
// FUNCTION
///

#define SEED_FUNC1 88271
__global__
void kernel_uniform_func1(double *dev_out, const double *dev_x, int M, int N, curandState *state)
{
	// Get position
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < M) {
		// Set seed
		curandState localState = state[i];
		double val = 0.0, x;
		for (int k = 0; k < N; ++k) {
			x = curand_uniform_double(&localState);
			val = val + x - x*dev_x[k];
		}
		dev_out[i] = val;
	}
}
__global__
void init_kernel_uniform(int M, int N, curandState *state)
{
	// Get position
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < M)
		curand_init(SEED_FUNC1, i, 0, &state[i]);
}

void func_uniform(int M, int N, const double *dev_x, double *dev_out)
{
	// Test dimension
	if (M < N) {
		printf("ERROR: WRong dimensions\n");
		exit(0);
	}
	kernel_uniform_func1<<< (M+255)/256, 256>>>(dev_out, dev_x, M, N, devStates);
}

#endif