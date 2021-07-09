#include <lmnslsqr/lmnslsqr.h>
#include <lmnslsqr/qmatrix.h>
#include <lmnslsqr/nslsqr.h>
#include <lmnslsqr/kernel.h>
#include <lmnslsqr/aux.h>
#include <curand.h>
#include <stdio.h>
#include <lmnslsqr/error.h>

#define MAX(X, Y) (((X) < (Y)) ? (Y) : (X))

////////////////////////////////////////////////////////////////
//////////////// UTILITY FUNCTIONS /////////////////////////////
////////////////////////////////////////////////////////////////

void _print_lmnslsqr_parameters(
	int M, int N, 
	double ini_factor, double min_factor, double red_factor, double inc_factor,
	double min_mu, double lower_mu, double upper_mu,
	int maxit, 
	double tol, double dtol, int dcount,
	int maxnslsqrit, int maxnslsqrrestart,
	double rtol, double stol, int siter, double ptol, int piter,
	int n_layers, int *bit_list, double qtol )
{
	fprintf( stderr, "\n---------------------------------------\n" );
	fprintf( stderr, "Report for initial parameters of lm-nsLSQR\n" );
	fprintf( stderr, "Problem dimension: %d x %d\n", M, N );
	fprintf( stderr, "Factors:\n" );
	fprintf( stderr, "\tInitial damping  : %.20f\n", ini_factor );
	fprintf( stderr, "\tMinimal damping  : %.20f\n", min_factor );
	fprintf( stderr, "\tReduction damping: %.20f\n", red_factor );
	fprintf( stderr, "\tIncrease damping : %.20f\n", inc_factor );
	fprintf( stderr, "\tMinimal mu       : %.20f\n", min_mu );
	fprintf( stderr, "\tLower bound mu   : %.20f\n", lower_mu );
	fprintf( stderr, "\tUpper bound mu   : %.20f\n", upper_mu );
	fprintf( stderr, "Iterations:\n" );
	fprintf( stderr, "\tlm-nsLSQR max. iterations: %d\n", maxit );
	fprintf( stderr, "\tnsLSQR max. iterations   : %d\n", maxnslsqrit );
	fprintf( stderr, "\tnsLSQR max. restarts     : %d\n", maxnslsqrrestart );
	fprintf( stderr, "Tolerances:\n" );
	fprintf( stderr, "\tlm-nsLSQR rel. residual tolerance : %.20f\n", tol );
	fprintf( stderr, "\tlm-nsLSQR solution diff. tolerance: %.20f\n", dtol );
	fprintf( stderr, "\tlm-nsLSQR solution diff. count    : %.20f\n", dcount );
	fprintf( stderr, "\tnsLSQR rel. residual tolerance    : %.20f\n", rtol );
	fprintf( stderr, "\tnsLSQR solution diff. tolerance   : %.20f\n", stol );
	fprintf( stderr, "\tnsLSQR solution diff. count       : %.20f\n", siter );
	fprintf( stderr, "\tnsLSQR saturation tolerance       : %.20f\n", ptol );
	fprintf( stderr, "\tnsLSQR saturation count           : %.20f\n", piter );
	fprintf( stderr, "Quantization:\n" );
	fprintf( stderr, "\tNumber of layers      : %d\n", n_layers );
	fprintf( stderr, "\tBits:" );
	for( int i = 0; i < n_layers; ++i )
		fprintf( stderr, " %d", bit_list[i] );
	fprintf( stderr, "\n" );
	fprintf( stderr, "\tQuantization tolerance: %.20f\n", qtol );
	fprintf( stderr, "\n---------------------------------------\n" );
}

void _print_memory_information(
	int M, int N, int nslsqrit, int *bit_list, int nlevels )
{
	int sum_bit = 0;
	for (int i = 0; i < nlevels; ++i)
		sum_bit = sum_bit + bit_list[i];
	double quant_memory = 
		1e-9*( (M*N*sum_bit)/8.0 + N*nlevels*8.0 + M*(32.0 + 64.0) + N*8.0 );
	double nslsqr_memory = 
		1e-9*8*(7*N + 4*M + nslsqrit*(N + M + 4*nslsqrit + 5) + 1);
	double solver_memory = 1e-9*8*(4*M + 5*N) + nslsqr_memory + quant_memory;
	fprintf( stderr, "\n---------------------------------------\n" );
	fprintf( stderr, "Report of memory used by lmnslsqr is:\n" );
	fprintf( stderr, "Quantization: %f\n", quant_memory );
	fprintf( stderr, "nsLSQR: %f\n", nslsqr_memory );
	fprintf( stderr, "lmnslsqr: %f\n", solver_memory );
	fprintf( stderr, "If the used memory is greater than the available memory, the program will be terminated.\n" );
	fprintf( stderr, "\n---------------------------------------\n" );
}

////////////////////////////////////////////////////////////////
//////////////////// LMILSQR implementation ////////////////////
////////////////////////////////////////////////////////////////


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
)
{
	if( debug ) {
		// Print parameter information
		_print_lmnslsqr_parameters(
			M, N, ini_factor, min_factor, red_factor, inc_factor, min_mu,
			lower_mu, upper_mu, maxit, tol, dtol, dcount, maxnslsqrit,
			maxnslsqrrestart, rtol, stol, siter, ptol, piter, n_layers, 
			bit_list, qtol );
		// Memory information
		_print_memory_information(M, N, maxnslsqrit, bit_list, n_layers);
	}

	// Setting seed
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, seed);

	// Evaluate initial guess
	double *dev_Fx0;
	cuda_check_error(
		cudaMalloc(&dev_Fx0, sizeof(double)*M),
		"lmnslsqr, allocation for Fxo\n"
	);
	func(M, N, dev_x0, dev_Fx0);

	// Compute initial residual
	double initial_residual;
	cublas_check_error(
		cublasDnrm2(*handle, M, dev_Fx0, 1, &initial_residual),
		"lmnslsqr, Computing the initial residual\n"
	);

	// LM factor or damping parameter
	double factor = ini_factor;

	// Iteration solution
	double *dev_xi;
	cuda_check_error(
		cudaMalloc(&dev_xi, sizeof(double)*N),
		"lmnslsqr, Allocation for xi\n"
	);
	cuda_check_error(
		cudaMemcpy(dev_xi, dev_x0, sizeof(double)*N, cudaMemcpyDeviceToDevice),
		"lmnslsqr, copy from x0 to xi\n"
	);

	// Vector to evaluate current solution
	double *dev_Fxi;
	double norm_Fxi;
	cuda_check_error(
		cudaMalloc(&dev_Fxi, sizeof(double)*M),
		"lmnslsqr, Allocation for Fxi\n"
	);
	// Initial evaluation
	func(M, N, dev_xi, dev_Fxi);
	cublas_check_error(
		cublasDnrm2(*handle, M, dev_Fxi, 1, &norm_Fxi), 
		"lmnslsqr, Computation of residual of current solution\n"
	);

	// Vector to use as initial guess, which will be equal to zero
	double *dev_ig;
	cuda_check_error(
		cudaMalloc(&dev_ig, sizeof(double)*N),
		"lmnslsqr, Allocation for the initial guess\n"
	);

	// Vector for LM step
	double *dev_dx;
	cuda_check_error(
		cudaMalloc(&dev_dx, sizeof(double)*N),
		"lmnslsqr, Allocation of the step\n"
	);

	// Vector for update candidate
	double *dev_y, *dev_Fy;
	cuda_check_error(
		cudaMalloc(&dev_y, sizeof(double)*N),
		"lmnslsqr, Allocation for candidate y\n"
	);
	cuda_check_error(
		cudaMalloc(&dev_Fy, sizeof(double)*M),
		"lmnslsqr, Allocation for F(y)\n"
	);

	// Auxiliar vectors
	double *dev_auxM, *dev_auxN;
	cuda_check_error(
		cudaMalloc(&dev_auxM, sizeof(double)*M),
		"lmnslsqr, Allocation for auxiliar vector 1\n"
	);
	cuda_check_error(
		cudaMalloc(&dev_auxN, sizeof(double)*N),
		"lmnslsqr, Allocation for auxiliar vector 2\n"
	);

	// Vector for prediction updates
	double ared, pred, norm1, gamma, norm2;

	// Quantization matrix structure
	struct qmatrix *qm = NULL;
	init_jac_approx(M, N);

	// Flag for quantization jacobian update
	int update_quantjac = 1;

	// double for alpha values
	double alpha;

	// Variables for computation of differecen tolerance
	double relstep = 1;
	int count = 0;
	int val_change = 1;

	// Index count for residual list
	int residual_index = 0;

	// Residual out for nslsqr
	double *res_nslsqr = (double *) malloc( 
		sizeof(double)*( maxnslsqrrestart*( maxnslsqrit + 1) ) );

	// Iteration of LM
	for (int k = 0; k < maxit; ++k) {
		if (debug)
			fprintf(stderr, "DEBUG: LM Iteration %d\n", k);

		if( debug )
			fprintf(stderr, "%.20f\n", norm_Fxi/initial_residual);

		// Tolerance test
		if (val_change) {
			if (relstep <= dtol)
				++count;
			else
				count = 0;
		}
		if (norm_Fxi/initial_residual < tol || count >= dcount) {
			if( debug ) {
				fprintf( stderr, "DEBUG: lm-nsLSQR finalized by: ");
				if( norm_Fxi/initial_residual < tol ) {
					fprintf( stderr, "rel. residual achieved\n" );
				}
				else {
					fprintf( stderr, "solution diff. tolerance achieved.\n" );
				}
			}
			k = maxit;
		}
		// Tolerance not achieved
		// Continue to the next iteration of LM
		else {
			// Compute the Quantization of hte Jacobian matrix
			if (update_quantjac) {
				// Evaluate current solution since it has changed
				func(M, N, dev_xi, dev_Fxi);
				cublas_check_error(
					cublasDnrm2(*handle, M, dev_Fxi, 1, &norm_Fxi), 
					"lmnslsqr, Computation of residual of current solution\n"
				);

				if( debug ) {
					fprintf( stderr, "DEBUG: New residua: %.20f\n",
						norm_Fxi/initial_residual );
				}

				// Store residual
				if (residual_out != NULL)
					residual_out[residual_index++] = norm_Fxi/initial_residual;
				
				qm = (struct qmatrix *) malloc(sizeof(struct qmatrix));
				init_qmatrix(qm, func, M, N, n_layers, bit_list, qtol, dev_xi, handle);
			}

			// Solve by nslsqr
			kernel_set_value<<< (N+255)/256, 256>>>(dev_ig, N, 0);
			if (debug)
				fprintf(stderr, "DEBUG: Calling nslsqr\n");
			
			nslsqr(
				qm, func, dev_xi, dev_ig, dev_Fxi, rtol, stol, siter, 
				ptol, piter, maxnslsqrit, maxnslsqrrestart, factor, 
				dev_dx, res_nslsqr, &prng, false );
			
			// Define prediction rates
			cuda_check_error(
				cudaMemcpy(dev_y, dev_xi, sizeof(double)*N,
					cudaMemcpyDeviceToDevice),
				"lmnslsqr, Copy from xi to y\n"
			);
			alpha = -1;
			cublas_check_error(
				cublasDaxpy(*handle, N, &alpha, dev_dx, 1, dev_y, 1), "lmnslsqr, first part of prediction rate\n"
			);
			
			func(M, N, dev_y, dev_Fy);
			
			// Actual reduction
			cublas_check_error(
				cublasDnrm2(*handle, M, dev_Fy, 1, &norm1), 
				"lmnslsqr, norm compuation\n"
			);
			ared = norm_Fxi*norm_Fxi - norm1*norm1;

			// Predicted reduction
			jac_approx(M, N, func, dev_xi, dev_dx, dev_auxM, eps, handle);
			cublas_check_error(
				cublasDnrm2(*handle, M, dev_auxM, 1, &norm1), 
				"lmnslsqr, norm computation\n"
			);
			cublas_check_error(
				cublasDdot(*handle, M, dev_Fxi, 1, dev_auxM, 1, &norm2), 
				"lmnslsqr, dot Fxi and auxM\n"
			);
			pred = 2*norm2 - norm1*norm1;
			gamma = ared/pred;

			// Compute the following LM damping factor
			if (debug)
				fprintf(stderr, "DEBUG: Starting damping factor update\n");
			
			if (gamma < min_mu) {
				update_quantjac = 0;
				factor = MAX(inc_factor*factor, min_factor);
				val_change = 0;
				if( debug )
					fprintf( stderr, "DEBUG: No update of damping factor\n" );
			}
			else if (min_mu <= gamma && gamma < lower_mu) {
				update_quantjac = 1;
				cuda_check_error(
					cudaMemcpy(dev_auxN, dev_y, sizeof(double)*N, 
						cudaMemcpyDeviceToDevice),
					"lmnslsqr, copy from y to auxN\n"
				);
				alpha = -1;
				cublas_check_error(
					cublasDnrm2(*handle, N, dev_auxN, 1, &norm1), 
					"lmnslsqr, norm in auxN\n"
				);
				cublas_check_error(
					cublasDaxpy(*handle, N, &alpha, dev_xi, 1, dev_auxN, 1), 
					"lmnslsqr, axpy xi and auxN\n"
				);
				cublas_check_error(
					cublasDnrm2(*handle, N, dev_auxN, 1, &norm2), 
					"lmnslsqr, norm computation in second case of gamma test\n"
				);
				relstep = norm2/norm1;
				cuda_check_error(
					cudaMemcpy(dev_xi, dev_y, sizeof(double)*N, 
						cudaMemcpyDeviceToDevice),
					"lmnslsqr, copy from y to xi\n"
				);
				factor = MAX(inc_factor*factor, min_factor);
				val_change = 1;
				if( debug ) {
					fprintf( stderr, "DEBUG: Damping factor increased\n" );
				}
			}
			else if (lower_mu <= gamma) {
				update_quantjac = 1;
				cuda_check_error(
					cudaMemcpy(dev_auxN, dev_y, sizeof(double)*N, 
						cudaMemcpyDeviceToDevice),
					"lmnslsqr, copy from y to auxN in lower_mu <= gamma\n"
				);
				alpha = -1;
				cublas_check_error(
					cublasDnrm2(*handle, N, dev_auxN, 1, &norm1), 
					"lmnslsqr, norm computation\n"
				);
				cublas_check_error(
					cublasDaxpy(*handle, N, &alpha, dev_xi, 1, dev_auxN, 1), 
					"lmnslsqr, axpy in xi\n"
				);
				cublas_check_error(
					cublasDnrm2(*handle, N, dev_auxN, 1, &norm2), 
					"lmnslsqr, norm of auxN\n"
				);
				relstep = norm2/norm1;
				cuda_check_error(
					cudaMemcpy(dev_xi, dev_y, sizeof(double)*N, 
						cudaMemcpyDeviceToDevice),
					"lmnslsqr, copy form y to xi in lower_mu <= gamma\n"
				);
				val_change = 1;
				if (upper_mu < gamma) {
					factor = red_factor*factor;
					if( debug ) {
						fprintf( stderr, "DEBUG: Damping factor decreased\n" );
					}
				}
				else if( debug ) {
					fprintf( stderr, "DEBUG: Damping factor maintained.\n" );
				}
			}
			if (factor < min_factor) {
				factor = min_factor;
				if( debug ) {
					fprintf( stderr, "DEBUG: Damping factor set to minimum\n" );
				}
			}

			// Free Quantziation matrix
			if (update_quantjac) {
				free_qmatrix(qm);
				free(qm);
				qm = NULL;
				if( debug ) {
					fprintf( stderr, "DEBUG: Need to update Jacobian\n" );
				}
			}
		}
	}

	// Add last element for marking
	if (residual_index <= maxit && residual_out != NULL)
		residual_out[residual_index] = -1;

	if( debug ) {
		fprintf( stderr, "DEBUG: Finalizing lm-nsLSQR\n" );
	}
	// Copy the solution to the output variable
	cudaMemcpy(dev_out, dev_xi, sizeof(double)*N, cudaMemcpyDeviceToDevice);
	curandDestroyGenerator(prng);
	free_jac_approx();
	// Free used memory
	free( res_nslsqr );
	cudaFree(dev_Fx0);
	cudaFree(dev_xi);
	cudaFree(dev_Fxi);
	cudaFree(dev_ig);
	cudaFree(dev_dx);
	cudaFree(dev_y);
	cudaFree(dev_Fy);
	cudaFree(dev_auxM);
	cudaFree(dev_auxN);
	if (qm != NULL) {
		free_qmatrix(qm);
		free(qm);
	}
}