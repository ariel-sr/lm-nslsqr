// Header file with our solver, called lmnslsqr or 
// Levenberg-Marquardt nsLSQR

// Memory used (bytes): (7N + 4M + maxnslsqrit*( N + M + 4*maxnslsqrit + 5 ) + 1)
// *8 + 8*(4M + 5N) + M*N*(b_1 + b_2 + ... + b_{n_layers})/8 + N*n_layers*8

#ifndef _LMNSLSQR_H
#define _LMNSLSQR_H

#include <cublas_v2.h>

// DEFAUT PARAMETERS
#define DEF_INI_FACTOR 1e-2
#define DEF_MIN_FACTOR 1e-8
#define DEF_RED_FACTOR 0.1
#define DEF_INC_FACTOR 10
#define DEF_MIN_MU 1e-4
#define DEF_LOWER_MU 0.25
#define DEF_UPPER_MU 0.75
#define DEF_TOL 1e-5
#define DEF_DTOL 1e-5
#define DEF_EPS 1e-5
#define DEF_DCOUNT 10
#define DEF_RTOL 1e-8
#define DEF_STOL 1e-8
#define DEF_SITER 20
#define DEF_PTOL 1e-8
#define DEF_PITER 20
#define DEF_QTOL 1e-5

#define DEFAULT_SEED 12396

/*
Solver function
Parameters:
	- func: A function pointer with the function to solve. 
			Must receive dimensions M, N, evaluation vector and output vector
	- M, N: Problem dimension
	- x0: Initial guess
	- ini_factor: Initial damping parameter for LM
	- min_factor: Minimum value for damping factor
	- red_factor: Reduction scaling factor for damping parameter.
	- inc_factor: Increase scaling factor for damping parameter
	- min_mu: Minimum value for ratio of predictions
	- lower_mu: Lower bound for ratio of predictions
	- upper_mu: Upper bound for raito of predictions
	- maxit: Max number of iterations of LM
	- tol: Relative residual tolerance for LM
	- dtol: A relative tolerance for difference of continuous solutions
	- dcount: How many continuous solutions to count for dtol
	- eps: Finite difference epsilon parameter
	- maxnslsqrit: Number of iteration per restart in nslsqr
	- maxnslsqrrestart: Number of restarts of nslsqr
	- rtol: Relative residual tolerance for nslsqr
	- stol: Progress residual for nslsqr
	- siter: How many points to consider for progress residual in nslsqr
	- ptol: Tolerance for saturation in nslsqr
	- piter: How many points to consider for saturation
	- n_layers: Number of layers for quantization
	- bit_list: Array of bits to use in each level
	- qtol: Quantization tolerance
	- handle: cublas Handle
	- dev_out: Device vector to store solution found by lmnslsqr
	- residual_out: A host vector to save residual computed at each iteration.
					Must be of size maxit.
	- seed: Seed to use for random parts in nslsqr.
*/
void lmnslsqr(
	void (*func)(int, int, const double *, double *),
	int M, 
	int N,
	const double *dev_x0,
	double ini_factor,
	double min_factor,
	double red_factor,
	double inc_factor,
	double min_mu,
	double lower_mu,
	double upper_mu,
	int maxit, 
	double tol, 
	double dtol,
	int dcount,
	double eps,
	int maxnslsqrit,
	int maxnslsqrrestart,
	double rtol,
	double stol,
	int siter,
	double ptol,
	int piter,
	int n_layers,
	int *bit_list,
	double qtol,
	cublasHandle_t *handle,
	double *dev_out,
	double *residual_out,
	int seed,
	bool debug
);

#endif