#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <curand.h>
#include <lmnslsqr/qmatrix.h>
#include <lmnslsqr/aux2.h>
#include <lmnslsqr/nslsqr.h>
#include <lmnslsqr/nslsqr_mf.h>
#include <lmnslsqr/error.h>
#include <lmnslsqr/aux.h>

// Global handle
cublasHandle_t handle;
// Curand state
curandState *devStates;

void (*func)(int, int, const double *, double *) = 0;

#include <func_dense1.h>
#include <func_dense2.h>
#include <func_sparse1.h>
#include <func_sparse2.h>
#include <func_normal.h>
#include <func_uniform.h>

__global__
void scaling_kernel(double *dev_out, double a, double b, int N)
{
	// Get position
	int k = blockIdx.x*blockDim.x + threadIdx.x;
	if (k < N) {
		dev_out[k] = a*dev_out[k] - b;
	}
}

void print_residual(double *error, int size)
{
	for (int i = 0; i < size; ++i) {
		if (error[i] != -1)
			printf("%.20f\n", error[i]);
		else
			i = size;
	}
}

void print_final_residual(int M, int N, double *dev_sol, double *dev_xi, double *dev_Fxi, double damp)
{
	double *dev_auxN1, *dev_auxN2, *dev_auxM;
	double norm1, norm2;
	cudaMalloc(&dev_auxN1, sizeof(double)*N);
	cudaMalloc(&dev_auxN2, sizeof(double)*N);
	cudaMalloc(&dev_auxM, sizeof(double)*M);

	printf("Final residual: ");
	// Compute (J^T*J + damp*I)*sol
	// Compute J*sol
	jac_approx(M, N, func, dev_xi, dev_sol, dev_auxM, 1e-6, &handle);
	jac_transpose(M, N, func, dev_xi, dev_auxM, dev_auxN1, 1e-6, &handle);
	cublasDaxpy(handle, N, &damp, dev_sol, 1, dev_auxN1, 1);
	// Comptue J^T*F
	jac_transpose(M, N, func, dev_xi, dev_Fxi, dev_auxN2, 1e-6, &handle);
	cublasDnrm2(handle, N, dev_auxN2, 1, &norm2);
	// Compute J^T*F - (J^T*J + damp*I)*sol = auxN2 - auxN1
	damp = -1;
	cublasDaxpy(handle, N, &damp, dev_auxN1, 1, dev_auxN2, 1);
	cublasDnrm2(handle, N, dev_auxN2, 1, &norm1);
	printf("%.20f\n", norm1/norm2);
	cudaFree(dev_auxN1);
	cudaFree(dev_auxN2);
	cudaFree(dev_auxM);
}

int main(int argc, char *argv[])
{
	// Problem format
	// M N n_layers *bits type problem

	// Read parameters
	int idx = 1;
	// Read dimension
	int M = atoi(argv[idx++]);
	int N = atoi(argv[idx++]);
	// Read number of layers
	int n_layers = atoi(argv[idx++]);
	// Read bits
	int *bits = (int *) malloc(sizeof(int)*n_layers);
	for (int i = 0; i < n_layers; ++i)
		bits[i] = atoi(argv[idx++]);
	// Read type
	int type = atoi(argv[idx++]);
	// Read problem
	int problem = atoi(argv[idx++]);

	// Rest of parameters
	// Read tolerance
	double tol = 1e-8;
	// Read siter
	int siter = 50;
	// Read piter
	int piter = 50;
	// Read maxiter
	int maxiter = 500;
	// Read restart
	int restart = 50;
	// Read damping factor
	double damp = 1e-5;

	// Cublas handle
	cublasCreate(&handle);
	// Random generator
	// Setting seed
	curandGenerator_t prng;
	int seed = 96625;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(prng, seed);
	
	// If problem 0 or 1, initialize random functions
	switch ( problem ) {
		case 0:
			// Curand for function call for random kernel function
			cuda_check_error(cudaMalloc((void **)&devStates, M * sizeof(curandState)), "main, generating devStates\n");
			init_kernel_normal<<< (M+255)/256, 256>>>(M, N, devStates);
			break;
		case 1:
			// Curand for function call for random kernel function
			cuda_check_error(cudaMalloc((void **)&devStates, M * sizeof(curandState)), "main, generating devStates\n");
			init_kernel_uniform<<< (M+255)/256, 256>>>(M, N, devStates);
			break;
		default:
			break;
	}

	// define problem
	switch ( problem )
	{
		// Normal problem
		case 0:
			func = &func_normal;
			break;
		// Uniform problem
		case 1:
			func = &func_uniform;
			break;
		// Sparse 1
		case 2:
			func = &func_sparse1;
			break;
		// sparse 2
		case 3:
			func = &func_sparse2;
			break;
		// dense 1
		case 4:
			func = &func_dense1;
			break;
		case 5:
			func = &func_dense2;
			break;
		default:
			break;
	}

	if( func )
	{
		// Intialize Jacobian function
		init_jac_approx(M, N);

		// Vector xi to eavluate Jacobian
		double *dev_xi, *xi;
		xi = (double *) malloc(sizeof(double)*N);
		cuda_check_error(cudaMalloc(&dev_xi, sizeof(double)*N), "main, Generating dev_xi\n");
		for (int i = 0; i < N; ++i)
			xi[i] = 0;
		curandGenerateUniformDouble(prng, dev_xi, N);
		scaling_kernel<<< (N+255)/256, 256>>>(dev_xi, 2, 1, N);

		// Initial guess
		double *dev_x0;
		cuda_check_error(cudaMalloc(&dev_x0, sizeof(double)*N), "main, generating dev_x0\n");

		// F(xi)
		double *dev_Fxi;
		cuda_check_error(cudaMalloc(&dev_Fxi, sizeof(double)*M), "main, generating F_xi\n");
		func(M, N, dev_xi, dev_Fxi);

		// Solution
		double *dev_sol;
		cuda_check_error(cudaMalloc(&dev_sol, sizeof(double)*N), "main, generating dev_sol\n");

		// Quantized matrix
		struct qmatrix *qm = NULL;

		// Parameters for iLSQR
		double rtol = tol;
		double stol = tol;
		double ptol = tol;
		
		// Residual list
		int size = restart*(maxiter + 1);
		double *res_list = (double *) malloc(sizeof(double)*size);

		// Run the experiment
		// Build Quantized matrix
		qm = (struct qmatrix *) malloc(sizeof(struct qmatrix));
		init_qmatrix(qm, func, M, N, n_layers, bits, 1e-12, dev_xi, &handle);
		fprintf(stderr, "Bit combination %d\n", 1);

		// Solve the least square problem
		cudaMemcpy(dev_x0, xi, sizeof(double)*N, cudaMemcpyHostToDevice);
		if (type == 0)
			nslsqr_mf(qm, func, dev_xi, dev_x0, dev_Fxi, rtol, stol, siter, ptol, piter, maxiter, restart, damp, dev_sol, res_list, &prng, true);
		else
			nslsqr(qm, func, dev_xi, dev_x0, dev_Fxi, rtol, stol, siter, ptol, piter, maxiter, restart, damp, dev_sol, res_list, &prng, true);

		// Print results
		print_final_residual(M, N, dev_sol, dev_xi, dev_Fxi, damp);
		
		// Free Quantized matrix
		free_qmatrix(qm);
		free(qm);

		// Free
		free(bits);
		free_jac_approx();
		cudaFree(dev_x0);
		cudaFree(dev_Fxi);
		cudaFree(dev_sol);
		free(res_list);
		cudaFree(dev_xi);
	}

	return 0;
}