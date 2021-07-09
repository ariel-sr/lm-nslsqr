// aux.cu
// Implementation file for the matrix-free product between a Jacobian matrix
// and a vector

#include <lmnslsqr/aux.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <lmnslsqr/kernel.h>
#include <lmnslsqr/error.h>

static double *dev_xp=NULL, *dev_xm=NULL, *dev_fp=NULL, *dev_fm=NULL;

// Initialization functions
void init_jac_approx(int M, int N)
{
	cudaMalloc(&dev_xp, sizeof(double)*N);
	cudaMalloc(&dev_xm, sizeof(double)*N);
	cudaMalloc(&dev_fp, sizeof(double)*M);
	cudaMalloc(&dev_fm, sizeof(double)*M);
}

void free_jac_approx()
{
	// Free data
	cudaFree(dev_xp);
	cudaFree(dev_xm);
	cudaFree(dev_fp);
	cudaFree(dev_fm);
}

// Matrix-free matrix-vector product
void jac_approx(
	int M,
	int N,
	void (*func)(int, int, const double *, double *),
	const double *dev_x,
	const double *dev_v,
	double *dev_out,
	double eps,
	cublasHandle_t *handle
)
{
	// Build vectors
	double alpha;
	
	// Make the finite differece calculations
	cudaMemcpy(dev_xp, dev_x, sizeof(double)*N, cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_xm, dev_x, sizeof(double)*N, cudaMemcpyDeviceToDevice);
	alpha = eps;
	cublas_check_error(
		cublasDaxpy(*handle, N, &alpha, dev_v, 1, dev_xp, 1),
		"aux, Jacobian-vector, sum\n");
	alpha = -eps;
	cublas_check_error(
		cublasDaxpy(*handle, N, &alpha, dev_v, 1, dev_xm, 1), 
		"aux, Jacobien-vector, substraction\n");
	func(M, N, dev_xp, dev_fp);
	func(M, N, dev_xm, dev_fm);

	// Store data in out vector
	cudaMemcpy(dev_out, dev_fp, sizeof(double)*M, cudaMemcpyDeviceToDevice);
	alpha = -1;
	cublas_check_error(
		cublasDaxpy(*handle, M, &alpha, dev_fm, 1, dev_out, 1), 
		"aux, Jacobien-vector, finite difference\n");
	alpha = 1/(2*eps);
	cublas_check_error(
		cublasDscal(*handle, M, &alpha, dev_out, 1), 
		"aux, Jacobian-vector, scaling by epsilon reciprocal\n");
}

//------------------------------------------------------------------------------
