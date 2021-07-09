// qmatrix.cu
// Implementation file for the quantization approximation.
// The structure implemented in this module is designed for the Jacobian
// matrix specifically. This implementation is constructed over the quant.h
// module, that abstract the process of a matrix being quantized given an
// specific number of parameters.

#include <lmnslsqr/quant.h>
#include <lmnslsqr/qmatrix.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <lmnslsqr/aux.h>
#include <lmnslsqr/kernel.h>
#include <lmnslsqr/error.h>

//////////////////////////////////////////////////////////////////
////////////////// KERNEL ////////////////////////////////////////
//////////////////////////////////////////////////////////////////

__global__
void kernel_set_int_value(int *dst, int M, int val)
{
	// Get position
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < M)
		dst[i] =  val;
}

__global__
void kernel_compute_quantization(int *dst, double *src, int M, double *D, int idx, double S)
{
	// Get position
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < M)
		dst[i] =  __double2int_rn(src[i]/D[idx] + S);
}

__global__
void kernel_compute_quantization_approx(double *dst, int *quant_vec, int M, double *D, int idx, double S)
{
	// Get position
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < M)
		dst[i] =  D[idx]*(quant_vec[i] - S);
}


//////////////////////////////////////////////////////////////////
////////////////// FUNCTIONS /////////////////////////////////////
//////////////////////////////////////////////////////////////////

void init_qmatrix(
	struct qmatrix *qmat,
	void (*func)(int, int, const double *, double *),
	int M, int N,
	int n_levels,
	int *bit_list,
	double norm_tol,
	const double *dev_x,
	cublasHandle_t *handle
)
{
	// Initialization process
	// Initialize simple data
	qmat->M = M;
	qmat->N = N;
	qmat->p = N/8;
	qmat->qmat_norm = 0;
	qmat->mat_norm = 0;
	qmat->error_mat_norm = 0;
	qmat->handle = handle;

	// Initialization of lists inside qmatrix
	qmat->n_levels = n_levels;
	qmat->dev_D_list = (double **) malloc(sizeof(double *)*qmat->n_levels);
	qmat->bit_list = (int *) malloc(sizeof(int)*qmat->n_levels);
	qmat->S_list = (double *) malloc(sizeof(double)*qmat->n_levels);
	qmat->qm = (struct quantization_matrix *) malloc(sizeof(struct quantization_matrix)*qmat->n_levels);
	for (int k = 0; k < qmat->n_levels; ++k) {
		cuda_check_error(cudaMalloc(&(qmat->dev_D_list[k]), sizeof(double)*qmat->N), "qmatrix, generating D_list\n");
		qmat->bit_list[k] = bit_list[k];
		qmat->S_list[k] = (1 << qmat->bit_list[k]-1) - 1;
		init_quantization_matrix(&(qmat->qm[k]), M, N, qmat->bit_list[k]);
	}

	// List of norms to track progress
	double *qmat_norm_list, *error_norm_list;
	qmat_norm_list  = (double *) malloc(sizeof(double)*n_levels);
	error_norm_list = (double *) malloc(sizeof(double)*n_levels);

	// Quantization process

	// Canonical vector
	double *dev_e;
	cuda_check_error(cudaMalloc(&dev_e, sizeof(double)*N), "qmatrix, generating dev_e\n");
	kernel_set_value<<<(qmat->N+255)/256, 256>>>(dev_e, N, 0);
	
	// Counter for number of columsn and current pack
	int count_cols = 0, current_pack = 0;

	// Auxiliary vectors to evaluate the function
	double *dev_aux;
	int *dev_current_aux;
	double *dev_middle_aux;
	double *dev_sum_aux;
	cuda_check_error(cudaMalloc(&dev_aux, sizeof(double)*M), "qmatrix, generating dev_aux\n");
	cuda_check_error(cudaMalloc(&dev_current_aux, sizeof(int)*M), "qmatrix, generating dev_current_aux\n");
	cuda_check_error(cudaMalloc(&dev_middle_aux, sizeof(double)*M), "qmatrix, generating dev_middle_aux\n");
	cuda_check_error(cudaMalloc(&dev_sum_aux, sizeof(double)*M), "qmatrix, generating dev_sum_aux\n");

	// Auxiliary scalars for norms
	double aux_norm, alpha;
	int idx;

	// List of Pack matrix -> M x 8
	int **dev_pack_matrix;
	dev_pack_matrix = (int **) malloc(sizeof(int *)*n_levels);
	for (int k = 0; k < n_levels; ++k) {
		cudaMalloc(&(dev_pack_matrix[k]), sizeof(int)*M*8);
		qmat_norm_list[k] = 0;
		error_norm_list[k] = 0;
	}

	// Auxiliary elements
	cuda_check_error(cudaMalloc(&(qmat->dev_pack_matrix), sizeof(int)*M*8), "qmatrix, generating dev_pack_matrix in qmat\n");
	cuda_check_error(cudaMalloc(&(qmat->dev_T), sizeof(double)*M*8), "qmatrix, generating dev_T in qmat\n");
	cuda_check_error(cudaMalloc(&(qmat->dev_aux), sizeof(double)*N), "qmatrix, generating dev_aux in qmat\n");

	// Iterate over each column
	for (int k = 0; k < N; ++k) {
		// Get the current column
		kernel_mod_value<<<1,1>>>(dev_e, k, 1);
		jac_approx(M, N, func, dev_x, dev_e, dev_aux, 1e-6, qmat->handle);
		kernel_mod_value<<<1,1>>>(dev_e, k, 0);
		// Set sum vector
		kernel_set_value<<<(M+255)/256, 256>>>(dev_sum_aux, M, 0);

		// Count columns
		++count_cols;

		// Compute the matrix norm
		cublas_check_error(cublasDnrm2(*qmat->handle, M, dev_aux, 1, &aux_norm), "qmatrix.cu, mat_norm computation\n");
		qmat->mat_norm = qmat->mat_norm + aux_norm*aux_norm;

		// Iterate over each level
		for (int i = 0; i < n_levels; ++i) {
			// Get the max value
			double P;
			cublas_check_error(cublasIdamax(*qmat->handle, M, dev_aux, 1, &idx), "qmatrix.cu, Max index computation\n");
			idx = idx-1; // cublas index starts from 1 due to FORTRAN
			cudaMemcpy(&P, dev_aux + idx, sizeof(double), cudaMemcpyDeviceToHost);
			if (P < 0)
				P = -P;

			// Discard if the value is close to zero
			if (P > 1e-12) {
				kernel_mod_value<<<1,1>>>(qmat->dev_D_list[i], k, P/qmat->S_list[i]);
				// Round values
				kernel_compute_quantization<<<(qmat->M+255)/256, 256>>>(dev_current_aux, dev_aux, M, qmat->dev_D_list[i], k, qmat->S_list[i]);
			}
			else {
				kernel_mod_value<<<1,1>>>(qmat->dev_D_list[i], k, 0);
			}

			// Compute norms
			kernel_compute_quantization_approx<<<(M+255)/256, 256>>>(dev_middle_aux, dev_current_aux, M, qmat->dev_D_list[i], k, qmat->S_list[i]);
			alpha = 1;
			cublas_check_error(cublasDaxpy(*qmat->handle, M, &alpha, dev_middle_aux, 1, dev_sum_aux, 1), "qmatrix.cu, axpy for norm of quantized matrix computation\n");
			cublas_check_error(cublasDnrm2(*qmat->handle, M, dev_sum_aux, 1, &aux_norm), "qmatrix.cu, computing norm of quantized column\n");
			qmat_norm_list[i] = qmat_norm_list[i] + aux_norm*aux_norm;
			alpha = -1;
			// The error vector is now the new aux to quantize
			cublas_check_error(cublasDaxpy(*qmat->handle, M, &alpha, dev_middle_aux, 1, dev_aux, 1), "qmatrix.cu, error column computation\n");
			cublas_check_error(cublasDnrm2(*qmat->handle, M, dev_aux, 1, &aux_norm), "qmatrix.cu, norm of error column\n");
			error_norm_list[i] = error_norm_list[i] + aux_norm*aux_norm;
			// Add the approximation column to pack matrix
			cudaMemcpy(dev_pack_matrix[i] + (count_cols - 1)*M, dev_current_aux, sizeof(int)*M, cudaMemcpyDeviceToDevice);
		}
		// Compress data
		if (count_cols == 8) {
			count_cols = 0;
			++current_pack;
			for (int j = 0; j < n_levels; ++j)
				compress_values(&(qmat->qm[j]), dev_pack_matrix[j], current_pack - 1);
		}
	}
	// Compute original norm
	qmat->mat_norm = sqrt(qmat->mat_norm);

	// Evaluate the achieved norm
	int used_levels = n_levels;
	for (int i = 0; i < n_levels; ++i) {
		if (sqrt(error_norm_list[i])/qmat->mat_norm < norm_tol) {
			used_levels = i+1;
			i = n_levels;
		}
	}
	// Get the number of elements that achieves the norm
	if (used_levels < n_levels) {
		qmat->n_levels = used_levels;
		int *new_bit_list = (int *) malloc(sizeof(int)*used_levels);
		double *new_S_list = (double *) malloc(sizeof(double)*used_levels);
		struct quantization_matrix *new_qm = (struct quantization_matrix *) malloc(sizeof(struct quantization_matrix)*used_levels);
		double **new_dev_D_list = (double **) malloc(sizeof(double *)*used_levels);
		// Copy data to use
		for (int i = 0; i < used_levels; ++i) {
			new_bit_list[i] = qmat->bit_list[i];
			new_S_list[i] = qmat->S_list[i];
			new_dev_D_list[i] = qmat->dev_D_list[i];
			new_qm[i] = qmat->qm[i];
		}
		// Clean simple data that not will be used
		free(qmat->bit_list);
		qmat->bit_list = new_bit_list;
		free(qmat->S_list);
		qmat->S_list = new_S_list;
		// Clean unused dev_D
		for (int i = used_levels; i < n_levels; ++i) {
			cudaFree(qmat->dev_D_list[i]);
			free_quantization_matrix(&(qmat->qm[i]));
		}
		free(qmat->dev_D_list);
		free(qmat->qm);
		qmat->dev_D_list = new_dev_D_list;
		qmat->qm = new_qm;
	}
	else if (used_levels > n_levels) {
		printf("Error: used_levels > n_levels\n");
	}
	// Compute norms
	qmat->qmat_norm      = sqrt(qmat_norm_list[used_levels-1]);
	qmat->error_mat_norm = sqrt(error_norm_list[used_levels-1]);

	// Free data
	free(qmat_norm_list);
	free(error_norm_list);
	cudaFree(dev_e);
	cudaFree(dev_aux);
	cudaFree(dev_current_aux);
	cudaFree(dev_sum_aux);
	cudaFree(dev_middle_aux);
	for (int k = 0; k < n_levels; ++k)
		cudaFree(dev_pack_matrix[k]);
	free(dev_pack_matrix);
}
void free_qmatrix(struct qmatrix *qmat)
{
	cudaFree(qmat->dev_pack_matrix);
	cudaFree(qmat->dev_aux);
	cudaFree(qmat->dev_T);
	free(qmat->bit_list);
	free(qmat->S_list);
	for (int i = 0; i < qmat->n_levels; ++i) {
		free_quantization_matrix(&(qmat->qm[i]));
		cudaFree(qmat->dev_D_list[i]);
	}
	free(qmat->qm);
	free(qmat->dev_D_list);
}

// Norm functions
// Compute norms functions
double qmatrix_norm(struct qmatrix *qmat)
{
	return qmat->qmat_norm;
}
double qmatrix_original_norm(struct qmatrix *qmat)
{
	return qmat->mat_norm;
}
double qmatrix_error_norm(struct qmatrix *qmat)
{
	return qmat->error_mat_norm;
}

int qmatrix_get_levels(struct qmatrix *qmat)
{
	return qmat->n_levels;
}

// Transpose-vector product
void qmatrix_tdot(struct qmatrix *qmat, double *dev_v, double *dev_y)
{
	// Constant for matrix-vector product
	double alpha = 1, beta = 0;
	// problem size
	int M = qmat->M, N = qmat->N;

	// Set values
	kernel_set_value<<<(N+255)/256, 256>>>(dev_y, N, 0);

	// Iterate over each level
	int current_col;
	for (int i = 0; i < qmat->n_levels; ++i) {
		current_col = 0;
		kernel_set_value<<<(N+255)/256, 256>>>(qmat->dev_aux, N, 0);
		for (int k = 0; k < qmat->p; ++k) {
			// Set the pack matrix to 0
			cudaMemset(qmat->dev_pack_matrix, 0, sizeof(int)*M*8);
			// get the matrix
			uncompress_values(&(qmat->qm[i]), k, qmat->dev_pack_matrix);	
			// Copy the matrix to the output matrix
			for (int j = 0; j < 8; ++j) {
				kernel_compute_quantization_approx<<<(M+255)/256, 256>>>(qmat->dev_T + M*j, qmat->dev_pack_matrix + j*M, M, qmat->dev_D_list[i], current_col, qmat->S_list[i]);
				++current_col;
			}
			// Compute the section from current_col-7 to current_col in the dot product
			cublas_check_error(cublasDgemv(*qmat->handle, CUBLAS_OP_T, M, 8, &alpha, qmat->dev_T, M, dev_v, 1, &beta, qmat->dev_aux+(current_col-8), 1), "qmatrix.cu, tdot, gemv, line 317\n");
		}
		// Sum with dev_aux with dev_y
		cublas_check_error(cublasDaxpy(*qmat->handle, N, &alpha, qmat->dev_aux, 1, dev_y, 1), "qmatrix.cu, tdot, axpy, line 320\n");
	}
}

