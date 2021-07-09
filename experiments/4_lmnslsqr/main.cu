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
#include <lmnslsqr/lmnslsqr.h>

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

// Constants for problem selection
#define NORMAL 0
#define UNIFORM 1
#define SPARSE1 2
#define SPARSE2 3
#define DENSE1 4
#define DENSE2 5

__global__
void scaling_kernel(double *dev_out, double a, double b, int N)
{
	// Get position
	int k = blockIdx.x*blockDim.x + threadIdx.x;
	if (k < N) {
		dev_out[k] = a*dev_out[k] - b;
	}
}

////////////////////////////////////////////////////////////////
//////////////////// UTILITY FUNCTIONS /////////////////////////
////////////////////////////////////////////////////////////////

void _print_initial_information(
	int M, int N, int n_layers, int* bits, int problem
)
{
	fprintf( stderr, "\n---------------------------------------\n" );
	fprintf( stderr, "Report of initial information\n" );
	fprintf( stderr, "Problem dimension: %d x %d\n", M, N );
	fprintf( stderr, "Number of layers: %d\n", n_layers );
	fprintf( stderr, "Bits:" );
	for( int i = 0; i < n_layers; ++i )
		fprintf( stderr, " %d", bits[i] );
	fprintf( stderr, "\n" );
	fprintf( stderr, "Problem type: %d\n", problem );
	fprintf( stderr, "\n---------------------------------------\n" );
}

void _print_final_reidual(double *error, int size)
{
	fprintf( stderr, "\n---------------------------------------\n" );
	fprintf( stderr, "Report of final residual:\n" );
	for (int i = 0; i < size; ++i) {
		if (error[i] != -1)
			fprintf( stderr,"%.20f\n", error[i]);
		else
			i = size;
	}
	fprintf( stderr, "\n---------------------------------------\n" );
}

////////////////////////////////////////////////////////////////
//////////////////// MAIN FUNCTION /////////////////////////////
////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
	// Use debug information?
	bool debug = true;

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
	// Read problem
	int problem = atoi(argv[idx++]);

	if( debug )
		_print_initial_information( M, N, n_layers, bits, problem );

	// Rest of parameters
	// Tolerances
	double tol = 1e-6;
	double dtol = 1e-10;
	double rtol = 1e-8;
	double stol = 1e-10;
	double ptol = 1e-10;
	double qtol = 1e-8;
	int dcount = 100;
	int siter = 30;
	int piter = 100;

	// Iterations
	int maxit = 10000;
	int nslsqr_maxit = 500;
	int nslsqr_restart = 20;

	// Cublas handle
	cublasCreate(&handle);

	// Random generator
	// Setting seed
	curandGenerator_t prng;
	int seed = 96625;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(prng, seed + problem);
	
	// If problem 0 or 1, initialize random functions
	switch ( problem ) {
		case NORMAL:
			// Curand for function call for random kernel function
			cuda_check_error(cudaMalloc((void **)&devStates, M * sizeof(curandState)), "main, generating devStates\n");
			init_kernel_normal<<< (M+255)/256, 256>>>(M, N, devStates);
			break;
		case UNIFORM:
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
		case NORMAL:
			func = &func_normal;
			break;
		// Uniform problem
		case UNIFORM:
			func = &func_uniform;
			break;
		// Sparse 1
		case SPARSE1:
			func = &func_sparse1;
			break;
		// sparse 2
		case SPARSE2:
			func = &func_sparse2;
			break;
		// dense 1
		case DENSE1:
			func = &func_dense1;
			break;
		case DENSE2:
			func = &func_dense2;
			break;
		default:
			break;
	}

	if( func )
	{
		// Initial guess for lm-nslsqr
		double *dev_x0;
		cuda_check_error(
			cudaMalloc(&dev_x0, sizeof(double)*N), 
			"main, Generating dev_x0\n"
		);
		curandGenerateUniformDouble(prng, dev_x0, N);
		scaling_kernel<<< (N+255)/256, 256>>>(dev_x0, 2, 1, N);

		// Solution
		double *dev_sol;
		cuda_check_error(
			cudaMalloc(&dev_sol, sizeof(double)*N), 
			"main, generating dev_sol\n"
		);

		// Residual list
		int size = maxit + 1;
		double *res_list = (double *) malloc(sizeof(double)*size);

		// Use the lmnslsqr method
		lmnslsqr(
			func, M, N, dev_x0, 
			DEF_INI_FACTOR, DEF_MIN_FACTOR, DEF_RED_FACTOR, DEF_INC_FACTOR,
			DEF_MIN_MU, DEF_LOWER_MU, DEF_UPPER_MU,
			maxit, tol, dtol, dcount, 1e-6, nslsqr_maxit, nslsqr_restart,
			rtol, stol, siter, ptol, piter, n_layers, bits, qtol,
			&handle, dev_sol, res_list, seed + problem, true );

		// Print results
		if( debug )
			_print_final_reidual( res_list, size );
		
		// Free
		cudaFree(dev_x0);
		cudaFree(dev_sol);
		free(res_list);
	}

	free(bits);
	return 0;
}