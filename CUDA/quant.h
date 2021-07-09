// quant.h
// File for Quantization structure
// This file contains functions and structure for quantization of a single
// matrix. Since the objective of the quantization is to compress data using
// an small number of bits, usually less than the minimum unity available in
// modern computers (8 bits), this module performs the 
// compression/uncompression to be able to store a matrix using effectively
// the desired number of bits.

// MEMORY USED (BYTES): M*N*b/8

#ifndef _QUANT_H
#define _QUANT_H 

#include <stdint.h>

struct quantization_matrix
{
	// M x N matrix
	int M;
	int N;
	// Bits and number of packs
	int b;
	int p;
	// compressed data.
	// This is an array of host pointers to device vectors of size M*b
	uint8_t **dev_comp_data;
	// A copy of a bintable
	int *dev_bintable;

};

// Set of functions to manage quantization procedure

// Init and clean quantization matrix
void init_quantization_matrix(struct quantization_matrix *qm, int M, int N, int b);
void free_quantization_matrix(struct quantization_matrix *qm);

// Compress/uncompress values
// Parameters:
// 	- qm: QuantizationMatrix structure
//	- dev_mat: Pointer to a matrix of size M x 8
//  - pos: Integer position of the matrix in a pack
void compress_values(struct quantization_matrix *qm, int *dev_mat, int pos);
void uncompress_values(struct quantization_matrix *qm, int pos, int *dev_mat);

#endif

