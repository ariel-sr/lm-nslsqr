// quant.cu
// Implementation file for a quantized matrix

#include <stdio.h>
#include <stdlib.h>
#include <lmnslsqr/quant.h>
#include <lmnslsqr/bintable.h>
#include <lmnslsqr/error.h>

//////////////////////////////////////////////////////////////////
////////////////// KERNEL ////////////////////////////////////////
//////////////////////////////////////////////////////////////////

// Apply a quantization procedure in parallel
// Parameters
//	- dst: Pointer to destination matrix of size Mxb
//	- src: source matrix of size Mx8
//	- dev_bintable: Binary table in device memory
//	- M, b: Number of equations and bits to use
__global__
void kernel_pack_compression(uint8_t *dst, int *src, int *dev_bintable, int M, int b)
{
	// Get position
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	// Iterate over each data in the i-th row
	if (i < M) {
		for (int j = 0; j < 8; ++j) {
			int data = src[i + M*j];
			for (int k = 0; k < b; ++k) {
				dst[i + k*M] += ((uint8_t) (1 << j))*dev_bintable[8*data + 7-k];
			}
		}
	}
}
//	- src: Pointer to compressed matrix of size Mxb
//	- dst: Destination matrix of size Mx8
//	- dev_bintable: Binary table in device memory
//	- M, b: Number of equations and bits to use
__global__
void kernel_pack_uncompression(uint8_t *src, int *dst, int *dev_bintable, int M, int b)
{
	// Get position
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	// Iterate over each datain the i-th row
	if (i < M) {
		// Itreate over each bit section in the source matrix
		for (int k = 0; k < b; ++k) {
			// get value to uncompress
			uint8_t value = src[i + k*M];
			for (int j = 0; j < 8; ++j) {
				dst[i + M*j] += ((uint8_t) (1 << k))*dev_bintable[8*value + 7-j];
			}
		}
	}
}

//////////////////////////////////////////////////////////////////
////////////////// Functions /////////////////////////////////////
//////////////////////////////////////////////////////////////////


// Initialize a quantization matrix structure
void init_quantization_matrix(struct quantization_matrix *qm, int M, int N, int b)
{
	qm->M = M;
	qm->N = N;
	qm->b = b;
	qm->p = N/8;
	qm->dev_comp_data = (uint8_t **) malloc(sizeof(uint8_t *)*qm->p);
	// Built the arrays of pointers
	for (int k = 0; k < qm->p; ++k) {
		cuda_check_error(cudaMalloc(&(qm->dev_comp_data[k]), sizeof(uint8_t)*qm->M*qm->b), "quant, Generating dev_comp_data\n");
		cudaMemset(qm->dev_comp_data[k], 0, sizeof(uint8_t)*qm->M*qm->b);
	}
	// Initialize inside bintable
	cuda_check_error(cudaMalloc(&(qm->dev_bintable), sizeof(int)*256*8), "quant, generating dev_bintable\n");
	cudaMemcpy(qm->dev_bintable, bintable, 256*8*sizeof(int), cudaMemcpyHostToDevice);
}
// Free memory used by a quantization structure
void free_quantization_matrix(struct quantization_matrix *qm)
{
	for (int k = 0; k < qm->p; ++k)
		cudaFree(qm->dev_comp_data[k]);
	free(qm->dev_comp_data);
	cudaFree(qm->dev_bintable);
}

// Compress/uncompress values
// Parameters:
// 	- qm: QuantizationMatrix structure
//	- dev_mat: Pointer to a device matrix of size M x 8
//  - pos: Integer position of the matrix in a pack
void compress_values(struct quantization_matrix *qm, int *dev_mat, int pos)
{
	// Test for a valid position
	if (pos < qm->p) {
		kernel_pack_compression<<<(qm->M+255)/256, 256>>>(qm->dev_comp_data[pos], dev_mat, qm->dev_bintable, qm->M, qm->b);
	}
}
void uncompress_values(struct quantization_matrix *qm, int pos, int *dev_mat)
{
	// Test for a valid position
	if (pos < qm->p) {
		kernel_pack_uncompression<<<(qm->M+255)/256, 256>>>(qm->dev_comp_data[pos], dev_mat, qm->dev_bintable, qm->M, qm->b);	
	}
}

