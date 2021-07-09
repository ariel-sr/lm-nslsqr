#ifndef _FUNC_SPARSE1_H
#define _FUNC_SPARSE1_H 

#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <lmnslsqr/kernel.h>

// KERNEL
__global__
void kernel_func_sparse1(double *dev_out, const double *dev_x, int M, int N)
{
	// Get position
	int k = blockIdx.x*blockDim.x + threadIdx.x;
	if (k < N-3) {
		double xi = dev_x[k], xi1 = dev_x[k+1], xi2 = dev_x[k+2], xi3 = dev_x[k+3];
		dev_out[6*k]   = xi + 3*xi1*(xi2 - 1) + pow(xi3, 2) - 1;
		dev_out[6*k+1] = pow(xi + xi1, 2) + pow(xi2 - 1, 2) - xi3 - 3;
		dev_out[6*k+2] = xi*xi1 - xi2*xi3;
		dev_out[6*k+3] = 2*xi*xi2 + xi1*xi3 - 3;
		dev_out[6*k+4] = pow(xi + xi1 + xi2 + xi3, 2) + pow(xi - 1, 2);
		dev_out[6*k+5] = xi*xi1*xi2*xi3 + pow(xi3 - 1, 2) - 1;
	}
}

// Sparse 1 function
void func_sparse1(int M, int N, const double *dev_x, double *dev_out)
{
	// Test dimension
	if (M < N || M != 6*(N-3)) {
		printf("ERROR: WRong dimensions\n");
		exit(0);
	}
	kernel_func_sparse1<<< ((N-3) + 255)/256, 256>>>(dev_out, dev_x, M, N);
}

#endif