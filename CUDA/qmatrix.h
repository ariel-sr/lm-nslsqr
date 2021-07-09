// qmatrix.h
// Header file of the Quantized Jacobian approximation
// Currently, the quantized approximation is designed to approximate
// matrices whose number of columns is a multiple of 8.
// This is for simplicity of the quantized implementation.

// Memory used (Bytes): M*N*(b_1 + b_2 + ... + b_{n_levels})/8 + N*n_levels*8

#ifndef _SQMATAPP_H
#define _SQMATAPP_H

#include <lmnslsqr/quant.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Struct used to represent a quantized Jacobian approximation
struct qmatrix
{
	// Matrix dimension
	//  - M: Number of rows
	//  - N: Number of columns
	int M;
	int N;

	// Number of packs in N
	// A pack represent the number of groups of 8 columns,
	// that is, p = N/8
	int p;
	
	// Norms
	// This norms may be computed during the construction of
	// the quantization process, so this saves computation for
	// later use of these norms.
	double qmat_norm;
	double mat_norm;
	double error_mat_norm;

	// Parameter list
	// Number of layers used for the quantized approximation
	int n_levels;
	// Number of bits use in each layer
	int *bit_list;
	// Vector of shift for linear transformation
	double *S_list;
	// A matrix containing the diagonal of scaling factors
	double **dev_D_list;
	// Quantization matrix structure
	struct quantization_matrix *qm;
	// Store the cublas handle
	cublasHandle_t *handle;
	// Tdot matrix and auxiliar elements
	int *dev_pack_matrix;
	double *dev_aux;
	double *dev_T;
};

// Initialization of quantized approximation
//
// Parameters:
//	- qmat: qmatrix to initialize
//	- func: Function pointer that computes the matrix-vector product between
//			the matri to quantize and an arbitrary vector. It is important
//			to note that the matrix computes everything in device to get
//			Maximum performance, so both pointers are to device memory.
//	- M, N: Dimension problem
//	- n_levels: Number of levens to use. This is the maximum number of levels to use
//	- bit_list: An array of bits to use. It's large must be n_levels
//	- norm_tol: Stopping tolerance for quantization
//	- x: the vector to evalaute the (matrix) function
//	- handle: cublas Handle
void init_qmatrix(
	struct qmatrix *qmat,
	void (*func)(int, int, const double *, double *),
	int M, int N,
	int n_levels,
	int *bit_list,
	double norm_tol,
	const double *x,
	cublasHandle_t *handle
);
// Free data of the structure
void free_qmatrix(struct qmatrix *qmat);

// Compute norms functions
double qmatrix_norm(struct qmatrix *sqmat);
double qmatrix_original_norm(struct qmatrix *sqmat);
double qmatrix_error_norm(struct qmatrix *sqmat);

// get number of levels used
int qmatrix_get_levels(struct qmatrix *qmat);

// Transpose-vector product
void qmatrix_tdot(struct qmatrix *qmat, double *dev_v, double *dev_out);

#endif