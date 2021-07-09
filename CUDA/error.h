// error.h
// File for error managmenet functions.
// If any error happens during a CUDA or CuBLAS function call, the program
// will print its respective error and will end its execution.

#ifndef _ERROR_H
#define _ERROR_H

#include <cublas_v2.h>

void cublas_check_error(cublasStatus_t stat, const char *msg);
void cuda_check_error(cudaError_t err, const char *msg);

#endif