#ifndef _FUNC_DENSE2_H
#define _FUNC_DENSE2_H 

#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <lmnslsqr/kernel.h>

// KERNEL
__global__
void kernel_func_dense2(double *dev_out, const double *dev_x, int M, int N, double sum)
{
	// Get position
	int k = blockIdx.x*blockDim.x + threadIdx.x;
	if (k < M) {
		//int p = (k+1)/N;
		//int i = (k+1) - p*N;
		int p = k/N;
		int i = k - p*N;
		double x = dev_x[i];
		dev_out[k] = pow(x, p+1)*log(sum + 1) + x;
	}
}

__global__
void kernel_square_dense2(double *dev_out, const double *dev_x, int M, int N)
{
	// Get position
	int k = blockIdx.x*blockDim.x + threadIdx.x;
	if (k < N) {
		double x = dev_x[k];
		dev_out[k] = x*x;
	}
}

// Sparse 1 function
void func_dense2(int M, int N, const double *dev_x, double *dev_out)
{
	// Test dimension
	if (M < N) {
		printf("ERROR: WRong dimensions\n");
		exit(0);
	}
	// Square vector
	kernel_square_dense2<<< (N+255)/256, 256>>>(dev_out, dev_x, M, N);
	// Sum vector
	double sum;
	cublasDasum(handle, N, dev_out, 1, &sum);
	kernel_func_dense2<<< (M + 255)/256, 256>>>(dev_out, dev_x, M, N, sum);
}

#endif