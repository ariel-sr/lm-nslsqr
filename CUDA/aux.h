// aux.h
// File with auxiliary routines.
// Specifically, routines to perform a matrix-free matrix-vector product
// between the Jacobian matrix of a function and a vector.
// The result of the product is computed using a second order finite-difference
//  like approach.

// Memory used (bytes) in jac_approx: 2M + 2N = 2(M+N)

#ifndef _AUX_H
#define _AUX_H

#include <cublas_v2.h>

// Matrix-free approximation of the jacobian matrix-vector product
// Parameters:
//	- M, N: problem size
// 	- func: POinter ot function. All its pointers parameters are device mem.
//  - dev_x: Pointer to device vector to evaluate the jacobian matrix
//	- dev_v: Pointer to device vector to multiply the matrix
//	- dev_out: Device memory to store the result
//	- eps: Epsilon for finite difference approximation
//	- handle: Cublas Handle pointer
void jac_approx(
	int M,
	int N,
	void (*func)(int, int, const double *, double *),
	const double *dev_x,
	const double *dev_v,
	double *dev_out,
	double eps,
	cublasHandle_t *handle
);

// Initialization functions for the routines jac_approx.
// Since the jacobian-vector product is used several times in nsLSQR, 
// these initialization routines were included to avoid the usage of
// malloc and cudacpy routines, which may produce a decrease in the performance
// of the overall method.
void init_jac_approx(int M, int N);
void free_jac_approx();

#endif