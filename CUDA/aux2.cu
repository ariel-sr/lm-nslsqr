// aux2.cu
// Implementation file for the matrix-free product between the
// transpose of the Jacobian matrix and a vector, using a
// of an orthogonal basis. The basis used are the canonical vectors

#include <lmnslsqr/aux.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <lmnslsqr/kernel.h>
#include <lmnslsqr/error.h>

void jac_transpose(
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
	// Canonical basis e and the jacobian times e
	double *dev_e, *dev_Je;
	double norm;
	cudaMalloc(&dev_e, sizeof(double)*N);
	cudaMalloc(&dev_Je, sizeof(double)*M);
	kernel_set_value<<< (N+255)/256, 256>>>(dev_e, N, 0);

	// Compute each element
	for (int i = 0; i < N; ++i) {
		kernel_mod_value<<<1,1>>>(dev_e, i, 1);
		jac_approx(M, N, func, dev_x, dev_e, dev_Je, eps, handle);
		kernel_mod_value<<<1,1>>>(dev_e, i, 0);
		cublasDdot(*handle, M, dev_v, 1, dev_Je, 1, &norm);
		kernel_mod_value<<<1,1>>>(dev_out, i, norm);
	}
	cudaFree(dev_e);
	cudaFree(dev_Je);
}
