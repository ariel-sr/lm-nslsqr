#ifndef _FUNC_SPARSE2_H
#define _FUNC_SPARSE2_H 

#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <lmnslsqr/kernel.h>

// KERNEL
__global__
void kernel_func_sparse2(double *dev_out, const double *dev_x, int M, int N)
{
	// Get position
	int k = blockIdx.x*blockDim.x + threadIdx.x;
	if (k < N/3) {
		dev_out[3*k]   = 0.6*dev_x[3*k] + 1.6*pow(dev_x[3*k], 3) - 7.2*pow(dev_x[3*k+1], 2) + 9.6*dev_x[3*k+1] - 4.8;
		dev_out[3*k+1] = 0.48*dev_x[3*k] - 0.72*pow(dev_x[3*k+1], 3) + 3.24*pow(dev_x[3*k+1], 2) - 4.32*dev_x[3*k+1] - dev_x[3*k+2] + 0.2*pow(dev_x[3*k+2], 3) + 2.16;
		dev_out[3*k+2] = 1.25*dev_x[3*k+2] - 0.25*pow(dev_x[3*k+2], 3);
	}
}

// Sparse 1 function
void func_sparse2(int M, int N, const double *dev_x, double *dev_out)
{
	// Test dimension
	if (M != N || N % 3 != 0) {
		printf("ERROR: WRong dimensions\n");
		exit(0);
	}
	kernel_func_sparse2<<< (N/3 + 255)/256, 256>>>(dev_out, dev_x, M, N);
}

#endif