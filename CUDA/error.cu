// errur.cu
// Implementation file for error.h, containing routines to manage errores
// or return values from the CUDA or CuBLAS functions.

#include <lmnslsqr/error.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

void cublas_check_error(cublasStatus_t stat, const char *msg)
{
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf(msg);
		switch (stat) {
			case CUBLAS_STATUS_ALLOC_FAILED:
				printf("CUBLAS_STATUS_ALLOC_FAILED\n");
				break;
			case CUBLAS_STATUS_INVALID_VALUE:
				printf("CUBLAS_STATUS_INVALID_VALUE\n");
				break;
			case CUBLAS_STATUS_ARCH_MISMATCH:
				printf("CUBLAS_STATUS_ARCH_MISMATCH\n");
				break;
			case CUBLAS_STATUS_MAPPING_ERROR:
				printf("CUBLAS_STATUS_MAPPING_ERROR\n");
				break;
			case CUBLAS_STATUS_EXECUTION_FAILED:
				printf("CUBLAS_STATUS_EXECUTION_FAILED\n");
				break;
			case CUBLAS_STATUS_INTERNAL_ERROR:
				printf("CUBLAS_STATUS_INTERNAL_ERROR\n");
				break;
			case CUBLAS_STATUS_NOT_SUPPORTED:
				printf("CUBLAS_STATUS_NOT_SUPPORTED\n");
				break;
			case CUBLAS_STATUS_LICENSE_ERROR:
				printf("CUBLAS_STATUS_LICENSE_ERROR\n");
				break;
			default:
				printf("Unknown error\n");
		}
		exit(1);
	}
}

void cuda_check_error(cudaError_t err, const char *msg)
{
	if (err != cudaSuccess) {
		printf(msg);
		printf("ERROR: %s\n", cudaGetErrorString(err));
		exit(0);
	}
}
