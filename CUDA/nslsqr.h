// nslsqr.h
// File with the declaration of the nsLSQR routine

// Memory used (bytes): (7N + 4M + maxit*( N + M + 4*maxit + 5 ) + 1)*8

#ifndef _NSLSQR_H
#define _NSLSQR_H 

#include <lmnslsqr/qmatrix.h>
#include <curand.h>

// nsLSQR routine. This solves || Fxi - Jx|| + \lambda||x||.
//
// Parameters:
//	- qm: Struct qmatrix that represent the quantization of the Jacobian matrix
//	- func: Nonlinear function. Receives and returns devices venctors
//	- dev_xi: Current solution in the Leveberg-Marquardt method
//	- dev_x0: Initial guess for nsLSQR
//	- dev_Fxi: Evaluation of F(xi). This saves one evaluation
//	- rtol: Tolerance for the relative residual of the nonlinear function
//	- stol: Progress tolerance
//	- siter: Number of iterations of non-progres termination
//  - ptol: Residual used for saturation tolerance
//  - piter: Number of past points to consider for residual saturation
// 	- maxit: Maximum number of inner nsLSQR iterations.
//	- restart: Number of restarts to use
// 	- damp: Regularization parameter of the nsLSQR method
//	- dev_out: Solution found by nslsqr
//  - residual_out: vector to store relative residual at each iteration. If null, it will not be used.
//	- prng: curand generator for vector perturbation
//  - debug: boolean to print information in the stderr
void nslsqr(
	struct qmatrix *qm,
	void (*func)(int, int, const double *, double *),
	const double *dev_xi,
	double *dev_x0,
	const double *dev_Fxi,
	double rtol,
	double stol,
	int siter,
	double ptol,
	int piter,
	int maxit,
	int restart,
	double damp,
	double *dev_out,
	double *residual_out,
	curandGenerator_t *prng,
	bool debug
);


#endif