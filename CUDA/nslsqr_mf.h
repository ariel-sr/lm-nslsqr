// nslsqr_mf.h
// File with the nsLSQR routine coupled with a matrix-free routine
// to compute the approximation of the transpose of the Jacobian
// matrix times a vector. The purpouse of this routine is to compare
// the proposed nsLSQR algorithm with LSQR.
// That is, this routine serves as a simulation of the LSQR for
// comparison purpouses.

// Memory used (bytes): (7N + 4M + maxit*( N + M + 4*maxit + 5 ) + 1)*8

#ifndef _NSLSQR_ARICA_H
#define _NSLSQR_ARICA_H 

#include <lmnslsqr/qmatrix.h>
#include <curand.h>

// nsLSQR routine
// Parameters:
//	- qm: Struct qmatrix that represent the quantization of the Jacobian matrix
//	- func: Nonlinear function. Receives and returns devices venctors
//	- dev_xi: Current solution in the Leveberg-Marquardt method
//	- dev_x0: Initial guess for nsLSQR
//	- dev_Fxi: Evaluation of F(xi). This saves one evaluation
//	- rtol: Tolerance for the relative residual of the nonlinear function
//	- stol: Slope tolerance
//	- lpoints: Number of points to compute stol
// 	- maxit: Maximum number of inner nsLSQR iterations.
//	- restart: Number of restarts to use
// 	- damp: Regularization parameter of the nsLSQR method
//	- dev_out: Solution found by nsLSQR
//  - residual_out: vector to store relative residual at each iteration. If null, it will not be used. It must be of size restart*(maxit + 1)
//	- prng: curand generator for vector perturbation
//  - debug: boolean to print information in the stderr
void nslsqr_mf(
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