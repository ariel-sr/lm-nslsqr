// aux2.h
// File with auxiliary routines.
// Specifically, routines to perform a matrix-free transpose matrix-vector
// product between the Jacobian matrix of a function and a vector.

// Memory used (bytes) in jac_approx: 2M + 2N = 2(M+N)
#ifndef _AUX2_H
#define _AUX2_H

#include <cublas_v2.h>

// Matrix-free approximation of the jacobian matrix-vector product
// Parameters:
//	- M, N: problem size
// 	- func: POinter ot function. All its pointers parameters are device mem.
// - dev_x: Pointer to device vector to evaluate the jacobian matrix
//	- dev_v: Pointer to device vector to multiply the matrix
//	- dev_out: Device memory to store the result
//	- eps: Epsilon for finite difference approximation
//	- handle: Cublas Handle pointer
void jac_transpose(
	int M,
	int N,
	void (*func)(int, int, const double *, double *),
	const double *dev_x,
	const double *dev_v,
	double *dev_out,
	double eps,
	cublasHandle_t *handle	
);

#endif