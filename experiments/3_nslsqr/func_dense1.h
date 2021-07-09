#ifndef _FUNC_DENSE1_H
#define _FUNC_DENSE1_H 

#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <lmnslsqr/kernel.h>

// KERNEL
__global__
void kernel_func_dense1(double *dev_out, const double *dev_x, int M, int N)
{
	// Get position
	int k = blockIdx.x*blockDim.x + threadIdx.x;
	if (k < M) {
		int p = k/N;
		int i = k - p*N;
		double s = 0.0;
		for (int j = 0; j < N; ++j)
			s = s + pow(cos(dev_x[j]), p+1);
		dev_out[k] = N - s + (k+1)*(1 - cos(dev_x[i])) - sin(dev_x[i]);
	}
}

// Sparse 1 function
void func_dense1(int M, int N, const double *dev_x, double *dev_out)
{
	// Test dimension
	if (M < N) {
		printf("ERROR: WRong dimensions\n");
		exit(0);
	}
	kernel_func_dense1<<< (M + 255)/256, 256>>>(dev_out, dev_x, M, N);
}

#endif