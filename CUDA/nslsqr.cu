// nslsqr.cu
// Implementation file for the nsLSQR method

#include <lmnslsqr/qmatrix.h>
#include <lmnslsqr/nslsqr.h>
#include <cublas_v2.h>
#include <lmnslsqr/aux.h>
#include <lmnslsqr/kernel.h>
#include <stdio.h>
#include <curand.h>
#include <lmnslsqr/error.h>
#include <math.h>


////////////////////////////////////////////////////////////////
//////////////////// KERNEL FUNCTIONS //////////////////////////
////////////////////////////////////////////////////////////////

// Auxiliary routine used by nsLSQR to scale the first row of a matrix src
// by scal, and store it in dst vector. The value of ld contains the leading
// dimension of src and the integer row represent which row is to be obtained
// from the matrix src.
__global__
void kernel_get_row(double *dst, double *src, int ld, int rows, int row, double scal)
{
	// Get position
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < rows)
	{
		dst[i] = scal*src[ld*i + row];
	}
}

////////////////////////////////////////////////////////////////
///////////////// Auxiliary functions //////////////////////////
////////////////////////////////////////////////////////////////

// Auxiliary routine used to compute the augmented product
// |   J   | * v
// | s*I_n |
// This routine is used to use nsLSQR for damping values different from zero
//
// Parameters
// - M: Number of rows of the Jacobian matrix
// - N: Number of columsn of the Jacobian matrix
// - func: Function used to perform Jacobian-vector product
// - dev_xi: Function to evaluate the jacobian matrix
// - dev_v: Function to perform the product
// - s: Scaling of the identity matrix
// - dev_dst: Vector to store the result
// - eps: Epsilon for finite difference
// - handle: Cublas handle
void augmented_dot(
	int M, 
	int N, 
	void (*func)(int, int, const double *, double *), 
	const double *dev_xi, 
	const double *dev_v, 
	double s,
	double *dev_dst,
	double eps,
	cublasHandle_t *handle
)
{
	// Perform first the product J*v
	jac_approx(M, N, func, dev_xi, dev_v, dev_dst, eps, handle);

	// Copy v into the lower part of dev_dst
	cuda_check_error(cudaMemcpy(dev_dst + M, dev_v, sizeof(double)*N, cudaMemcpyDeviceToDevice), "augmented_dot, Copy lower part of dev_dst\n");
	// Scaling the lower part of dev_dst -> s*v
	cublas_check_error(cublasDscal(*handle, N, &s, dev_dst + M, 1), "augmented_dot, Scaling lower part of dev_dst\n");
}

// Auxiliary routine used to compute the transpose of the augmented product
// |   J   |^T * w
// | s*I_n |
// This routine is used to use nsLSQR for damping values different from zero
//
// Parameters
// - M: Number of rows of the Jacobian matrix
// - N: Number of columsn of the Jacobian matrix
// - qm: Struct of the quantized matrix
// - dev_w: Vector for product. Size M + N
// - s: scaling of the identity matrix
// - dev_dst: Vector to store the result. Size N
void augmented_tdot(
	int M,
	int N,
	struct qmatrix *qm,
	double *dev_w,
	double s,
	double *dev_dst
)
{
	// Perform the \tilde{J}^T * w[0:M]. That is, the first part of the product
	qmatrix_tdot(qm, dev_w, dev_dst);
	// Now we need to compute dev_dst := dev_dst + s*w[M+1:M+N]
	cublas_check_error(cublasDaxpy(*qm->handle, N, &s, dev_w+M, 1, dev_dst, 1), "augmented_tdot, axpy for scaling the identity matrix\n");
}

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
//  - ptol: Residual progress tolerance
//  - piter: Number of past points to consider for residual progress
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
)
{
	// Problem dimension
	int N = qm->N;
	// Since this is an augmented problem, dimension is m + n
	int M = qm->M + N;

	// Square root of damping factor
	double sdamp = sqrt(damp);

	// Initial residuals
	double initial_r0 = -1, initial_hat_r0 = -1;
	// Alpha for Daxpy
	double alpha, beta, gamma;
	// Auxiliary values
	double norm_r0, norm_hat_r0;

	// Create a residual list if its NULL
	bool delete_residual = false;
	if ( residual_out == NULL) {
		delete_residual = true;
		residual_out = (double *) malloc( 
			sizeof(double)*(restart*(maxit + 1)) );
	}

	// Termination flag
	int flag = 0;

	// Build new augmented dev_Fxi
	double *dev_big_Fxi;
	cuda_check_error(cudaMalloc(&dev_big_Fxi, sizeof(double)*M), "nslsqr, allocation of big_Fxi memory\n");
	// Initialize it to zero
	kernel_set_value<<< (M+255)/256, 256>>>(dev_big_Fxi, M, 0);
	// Copy Fxi to upper part of big_Fxi
	cuda_check_error(cudaMemcpy(dev_big_Fxi, dev_Fxi, sizeof(double)*qm->M, cudaMemcpyDeviceToDevice), "nslsqr, Copy upper part of big_Fxi\n");

	// Residuals and solution of each restart
	double *dev_r0, *dev_hat_r0, *dev_x;
	cuda_check_error(cudaMalloc(&dev_r0, sizeof(double)*N), "nslsqr, allocation of r0 memory\n");
	cuda_check_error(cudaMalloc(&dev_hat_r0, sizeof(double)*M), "nslsqr, allocation for hatr0 memory\n");
	cuda_check_error(cudaMalloc(&dev_x, sizeof(double)*N), "nslsqr, allocation of memory for x\n");
	cudaMemcpy(dev_x, dev_x0, sizeof(double)*N, cudaMemcpyDeviceToDevice);

	// Index for residual list
	int residual_index = 0;

	// Auxiliary vectors
	double *dev_auxN, *dev_auxN2, *dev_auxM;
	cuda_check_error(cudaMalloc(&dev_auxN, sizeof(double)*N), "nslsqr, allocation of memory for auxN\n");
	cuda_check_error(cudaMalloc(&dev_auxN2, sizeof(double)*N), "nslsqr, allocation of memory for auxN2\n");
	cuda_check_error(cudaMalloc(&dev_auxM, sizeof(double)*M), "nslsqr, allocation of memory for auxM\n");

	// Build U and V matrices
	double *dev_U, *dev_V;
	cuda_check_error(cudaMalloc(&dev_V, sizeof(double)*N*maxit), "nslsqr, allocation of memory for nslsqr V matrix\n");
	cuda_check_error(cudaMalloc(&dev_U, sizeof(double)*M*maxit), "nslsqr, allocation of memory for nslsqr U matrix\n");

	// Matrix B
	double *dev_B;
	cuda_check_error(cudaMalloc(&dev_B, sizeof(double)*(maxit+1)*maxit), "nslsqr, allocation of memory for nslsqr B matrix\n");

	// Matrices for QR Decomposition
	double *dev_Q, *dev_R, *dev_auxQ, *dev_auxR;
	cuda_check_error(cudaMalloc(&dev_Q, sizeof(double)*(maxit+1)*maxit), "nslsqr, allocation of memory for nslsqr-QR Q matrix\n");
	cuda_check_error(cudaMalloc(&dev_auxR, sizeof(double)*(maxit+1)), "nslsqr, allocation of memory for auxR\n");
	cuda_check_error(cudaMalloc(&dev_auxQ, sizeof(double)*maxit), "nslsqr, allocation of memory for auxQ\n");
	cuda_check_error(cudaMalloc(&dev_R, sizeof(double)*maxit*maxit), "nslsqr, allocation of memory for nslsqr-QR R matrix\n");

	// Some perturbation vectors
	double *dev_eps;
	cuda_check_error(cudaMalloc(&dev_eps, sizeof(double)*N), "nslsqr, allocation of memory for perturbation vector\n");

    // Slope for residual decrease test
    int no_progress = 0;
    // Progres of residual
    double slope = 1;
    
	// Iterate over each restart
	for (int i = 0; i < restart; ++i) {
		if (debug)
			fprintf(stderr, "DEBUG: Restart i = %d\n", i);
		// Use initial guess for current residual
		// Current problem is || Fxi - J*(x0 + xk)||
		// We will compute hr0 = Fxi - Jx0
		// Compute hr0 = Jx0 first
		augmented_dot(qm->M, qm->N, func, dev_xi, dev_x0, sdamp, dev_hat_r0, 1e-6, qm->handle);

		// Compute hr0 = Fxi - hr0 = -1*(-Fxi + hr0)
		alpha = -1;
		cublas_check_error(cublasDaxpy(*qm->handle, M, &alpha, dev_big_Fxi, 1, dev_hat_r0, 1), "nslsqr, axpy for initial residual hatr0\n");
		cublas_check_error(cublasDscal(*qm->handle, M, &alpha, dev_hat_r0, 1), "nslsqr, correction of hatr0\n");

		// New problem is || r0 - Jxk ||

		// Compute residual r0 = J^T * hr0
		augmented_tdot(qm->M, qm->N, qm, dev_hat_r0, sdamp, dev_r0);

		// Compute residuals
		cublas_check_error(cublasDnrm2(*qm->handle, M, dev_hat_r0, 1, &norm_hat_r0), "nslsqr, norm of hatr0\n");
		cublas_check_error(cublasDnrm2(*qm->handle, N, dev_r0, 1, &norm_r0), "nslsqr, norm of r0\n");

		// Set initial norms
		if (initial_r0 < 0) {
			initial_r0 = norm_r0;
			initial_hat_r0 = norm_hat_r0;
		}

		// Store initial residual
		if (residual_out != NULL) {
			residual_out[residual_index++] = norm_hat_r0/initial_hat_r0;
			if (debug)
				fprintf(stderr, "DEBUG: initial nsLSQR residual: %.20f\n", residual_out[residual_index-1]);
		}

		// Initial vector U and V
		cuda_check_error(cudaMemcpy(dev_U, dev_hat_r0, sizeof(double)*M, cudaMemcpyDeviceToDevice), "nslsqr, copy hatr0 to first U\n");
		alpha = 1/norm_hat_r0;
		cublas_check_error(cublasDscal(*qm->handle, M, &alpha, dev_U, 1), "nslsqr, normalizing first U\n");
		cuda_check_error(cudaMemcpy(dev_V, dev_r0, sizeof(double)*N, cudaMemcpyDeviceToDevice), "nslsqr, copy r0 to first V\n");
		alpha = 1/norm_r0;
		cublas_check_error(cublasDscal(*qm->handle, N, &alpha, dev_V, 1), "nslsqr, normalizing first V\n");

		// Iteration of each restart
		for (int k = 0; k < maxit; ++k) {
			// First part of LSQR: Compute next U by Gram-Schmidt
			augmented_dot(qm->M, qm->N, func, dev_xi, dev_V + N*k, sdamp, dev_auxM, 1e-6, qm->handle);
			
			// Set current B column to zero
			kernel_set_value<<< (maxit+1+255)/256, 256>>>(dev_B + (maxit+1)*k, maxit+1, 0);
			
			// Gram-Schmidt process
			for (int j = 0; j < k+1; ++j) {
				cublas_check_error(cublasDdot(*qm->handle, M, dev_auxM, 1, dev_U + M*j, 1, &alpha), "nslsqr, Scalar computation in first Gram-Schmidt\n");
				kernel_mod_value<<<1, 1>>>(dev_B + (maxit+1)*k, j, alpha);
				alpha = -alpha;
				cublas_check_error(cublasDaxpy(*qm->handle, M, &alpha, dev_U + M*j, 1, dev_auxM, 1), "nslsqr, First Gram-Schmidt modification for stability\n");
			}
			cublas_check_error(cublasDnrm2(*qm->handle, M, dev_auxM, 1, &alpha), "nslsqr, norm compuation of last element in current column of B\n");
			
			// Test if the solution was found
			if (alpha > 1e-12) {
				// Store the computed U vector and the last value of B
				kernel_mod_value<<<1, 1>>>(dev_B + (maxit+1)*k, k+1, alpha);
				if (debug)
					fprintf(stderr, "DEBUG: b_{%d, %d} = %.20f\n", k+1, k, alpha);
				
				// If there are iteration left, build the next U and V vector
				if (k < maxit-1) {
					alpha = 1/alpha;
					cublas_check_error(cublasDscal(*qm->handle, M, &alpha, dev_auxM, 1), "nslsqr, normalization of new U\n");
					cudaMemcpy(dev_U + M*(k+1), dev_auxM, sizeof(double)*M, cudaMemcpyDeviceToDevice);

					// Flags and perturbation vectors for premature break in quantization
					int continue_V_vector = 1;

					// Next V vector
					augmented_tdot(qm->M, qm->N, qm, dev_U + M*(k+1), sdamp, dev_auxN2);
					kernel_set_value<<< (N+255)/256, 256>>>(dev_eps, N, 0);
					while (continue_V_vector) {
						// Second part of LSQR and begin of nsLSQR
						// Compute next vector V by Gram-Schmidt and quantization

						// Add the perturbation vector for avoid breakdown
						alpha = 1;
						cudaMemcpy(dev_auxN, dev_auxN2, sizeof(double)*N, cudaMemcpyDeviceToDevice);
						cublas_check_error(cublasDaxpy(*qm->handle, N, &alpha, dev_eps, 1, dev_auxN, 1), "nslsqr, adding a small perturbation for breakdown\n");

						// Gram-Schmidt
						for (int j = 0; j < k+1; ++j) {
							cublas_check_error(cublasDdot(*qm->handle, N, dev_auxN, 1, dev_V + N*j, 1, &alpha), "nslsqr, scalar computation in second Gram-Schmidt\n");
							alpha = -alpha;
							cublas_check_error(cublasDaxpy(*qm->handle, N, &alpha, dev_V + N*j, 1, dev_auxN, 1), "nslsqr, second Gram-Schmidt modification for stability\n");
						}
						cublas_check_error(cublasDnrm2(*qm->handle, N, dev_auxN, 1, &alpha), "nslsqr, norm computation of resulting vector in second Gram-Schmidt\n");

						// if alpha is too small, it means a premature breakdown
						// has occured due to a non-fullrank quantized matrix
						// Solution: Add a small perturbation
						if (alpha > 1e-12) {
							if (debug)
								fprintf(stderr, "DEBUG: beta_{%d, %d} = %.20f\n", k+1, k+1, alpha);
							alpha = 1/alpha;
							cublas_check_error(cublasDscal(*qm->handle, N, &alpha, dev_auxN, 1), "nslsqr, normalization of new V\n");
							cudaMemcpy(dev_V + N*(k+1), dev_auxN, sizeof(double)*N, cudaMemcpyDeviceToDevice);
							continue_V_vector = 0;
						}
						else {
							if (debug)
								fprintf(stderr, "DEBUG: A zero value in generation of V: adding a small perturbation\n");
							curandGenerateUniformDouble(*prng, dev_eps, N);
						}
					}
				}
			}
			else {
				flag = 1;
			}

			// Solve least-square by proposed QR decomposition
			// If it is the first iteration, we need to compute initial QR decomposition
			if (k == 0) {
				cudaMemcpy(dev_auxN, dev_B, sizeof(double)*2, cudaMemcpyDeviceToDevice);
				cublas_check_error(cublasDnrm2(*qm->handle, 2, dev_auxN, 1, &alpha), "nslsqr, QR, norm of first R\n");
				kernel_set_value<<< (maxit+255)/256, 256 >>>(dev_R, maxit, 0);
				kernel_mod_value<<<1,1>>>(dev_R, 0, alpha);
				kernel_set_value<<< ((2*maxit+1)+255)/256, 256 >>>(dev_Q, maxit+1, 0);
				alpha = 1/alpha;
				cublas_check_error(cublasDaxpy(*qm->handle, 2, &alpha, dev_auxN, 1, dev_Q, 1), "nslsqr, QR, compuation of first Q\n");
			}
			// Rest of QR decomposition
			else {
				// Compute new R
				// Fill new R column with zeroes
				kernel_set_value<<< (maxit+255)/256, 256 >>>(dev_R + maxit*k, maxit, 0);
				// Fill with zeroes the new column to perform QR
				kernel_set_value<<< ((maxit+1)+255)/256, 256 >>>(dev_auxR, maxit+1, 0);
				// Set first values with new column of B
				cudaMemcpy(dev_auxR, dev_B+(maxit+1)*k, sizeof(double)*(k+2), cudaMemcpyDeviceToDevice);
				
				// Perform the QR computation
				for (int j = 0; j < k; ++j) {
					cublas_check_error(cublasDdot(*qm->handle, k+2, dev_auxR, 1, dev_Q+(maxit+1)*j, 1, &alpha), "nslsqr, QR, dot in Gram-Schmidt\n");
					kernel_mod_value<<<1,1>>>(dev_R + maxit*k, j, alpha);
					alpha = -alpha;
					cublas_check_error(cublasDaxpy(*qm->handle, k+2, &alpha, dev_Q+(maxit+1)*j, 1, dev_auxR, 1), "nslsqr, QR, modification of Gram-Schmidt for stability\n");
				}
				// Set new value for R
				cublas_check_error(cublasDnrm2(*qm->handle, k+2, dev_auxR, 1, &alpha), "nslsqr, QR, compute new R\n");
				kernel_mod_value<<<1,1>>>(dev_R+maxit*k, k, alpha);

				// Set new value of Q
				kernel_set_value<<<(maxit+1+255)/256, 256>>>(dev_Q + (maxit+1)*k, maxit+1, 0);
				alpha = 1/alpha;
				cublas_check_error(cublasDscal(*qm->handle, k+2, &alpha, dev_auxR, 1), "nslsqr, QR, normalizing new Q\n");
				cudaMemcpy(dev_Q + (maxit+1)*k, dev_auxR, sizeof(double)*(k+2), cudaMemcpyDeviceToDevice);
			}

			// Solve the least-square problem
			kernel_get_row<<<((k+1)+255)/256, 256>>>(dev_auxQ, dev_Q, maxit+1, k+2, 0, norm_hat_r0);
						
			// Solve lower triangular linear system and store it dev_auxQ
			cublas_check_error(cublasDtrsv(*qm->handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, k+1, dev_R, maxit, dev_auxQ, 1), "nslsqr, solving small least-square (triangular system)\n");

			// Compute the next solution
			alpha = 1;
			beta = 0;
			cudaMemcpy(dev_auxN, dev_x, sizeof(double)*N, cudaMemcpyDeviceToDevice);
			cublas_check_error(cublasDgemv(*qm->handle, CUBLAS_OP_N, N, k+1, &alpha, dev_V, N, dev_auxQ, 1, &beta, dev_x, 1), "nslsqr, computing new approximate solution\n");

			// Measure the new residual hr0 - Jx_k = hr0 - dev_auxN = -(-hr0 + dev_auxN)
			augmented_dot(qm->M, qm->N, func, dev_xi, dev_x, sdamp, dev_auxM, 1e-6, qm->handle);
			alpha = -1;
			cublas_check_error(cublasDaxpy(*qm->handle, M, &alpha, dev_hat_r0, 1, dev_auxM, 1), "nslsqr, axpy new norm\n");
			cublas_check_error(cublasDscal(*qm->handle, M, &alpha, dev_auxM, 1), "nslsqr, scaling new norm\n");
			cublas_check_error(cublasDnrm2(*qm->handle, M, dev_auxM, 1, &alpha), "nslsqr, norm in new norm\n");
			alpha = alpha/initial_hat_r0;
			if (residual_out != NULL) {
				residual_out[residual_index++] = alpha;
				if (debug)
					fprintf(stderr, "DEBUG: nsLSQR residual: %.20f\n", residual_out[residual_index-1]);
			}
			
			// Test progress computation
			cublas_check_error( cublasDnrm2(*qm->handle, N, dev_x, 1, &beta) , "nslsqr, Norm of current solution");
			gamma = -1;
			cublas_check_error( cublasDaxpy(*qm->handle, N, &gamma, dev_x, 1, dev_auxN, 1), "nslsqr, difference of solutions");
			cublas_check_error( cublasDnrm2(*qm->handle, N, dev_auxN, 1, &gamma) , "nslsqr, Norm of difference of solutions");
			if (gamma/beta <= stol)
				no_progress++;
			else
				no_progress = 0;

			// Progress in residual
			slope = 1;
			if (i*restart + k + 1 >= piter) {
				// Make linear regression for slope estimation
				// Since is a linear regression, the normal equation is a
				// 2x2 linear system easy to solve by direct Kramer's rule
				double Sy = 0, Sx = 0, Sx2 = 0, Sxy = 0;
				Sx = piter*(piter + 1)/2;
				Sx2 = piter*(piter + 1)*(2*piter + 1)/6;
				for (int j = 0; j < piter; ++j) {
					Sy = Sy + log10(residual_out[residual_index-1-j]);
					Sxy = Sxy + log10(residual_out[residual_index-1-j])*(piter-j);
				}
				slope = (piter*Sxy - Sx*Sy)/(piter*Sx2 - Sx*Sx);
			}

			// Test for tolerance accomplishment
			if (alpha <= rtol || no_progress == siter || flag || slope*slope <= ptol*ptol) {
				k = maxit;
				i = restart;
			}
		}
		alpha = 1;
		cublas_check_error(cublasDaxpy(*qm->handle, N, &alpha, dev_x, 1, dev_x0, 1), "nslsqr, add solution found to initial guess\n");
	}
	cudaMemcpy(dev_out, dev_x0, sizeof(double)*N, cudaMemcpyDeviceToDevice);

	// Add last element for marking
	if (residual_index != restart*(maxit + 1) && residual_out != NULL)
		residual_out[residual_index] = -1;

	// Free data
	if ( delete_residual )
		free(residual_out);
	cudaFree(dev_big_Fxi);
	cudaFree(dev_r0);
	cudaFree(dev_hat_r0);
	cudaFree(dev_auxN);
	cudaFree(dev_auxN2);
	cudaFree(dev_auxM);
	cudaFree(dev_U);
	cudaFree(dev_V);
	cudaFree(dev_B);
	cudaFree(dev_Q);
	cudaFree(dev_R);
	cudaFree(dev_auxQ);
	cudaFree(dev_auxR);
	cudaFree(dev_eps);
	cudaFree(dev_x);
}