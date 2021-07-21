# Summary

This repository contains the implementations and codes of my Master's degree thesis, called "A Limited-memory Levenberg-Marquardt algorithm for solving large-scale nonlinear least-square problems".

# Content
The repository contains several folders. The explanation of each folder is given depending on its purpose.

## Code and implementations

* CUDA: This folder contains the Quantization approximation code, the nsLSQR code and the lm-nsLSQR code. Also, it contains the necessary routines to execute nsLSQR as LSQR, with a matrix-free approximation of the transpose of a matrix times a vector. The modules in the folder are the following:
	* aux: Auxiliary routines for nsLSQR. In particular, contains the code to approximate the product between the Jacobian matrix and a vector using finite-difference.
	* aux2: Contains the code for a matrix-free approximation of the product between the transpose of the Jacobian matrix and a vector. It is used to make nsLSQR to behave like LSQR.
	* bintable.h: A static two-dimensional array, containing the bit representation of each positive integre from 0 to 255, in 8 bits. It is used in the quantized approximation.
	* error: Contains auxiliary routines to obtain information of CUDA or cublas errors.
	* kernel: Contains some CUDA kernels that are used in the nsLSQR code.
	* lmnslsqr: Code of the lm-nsLSQR (Levenberg-Marquardt solver with nsLSQR).
	* nslsqr: Code of nsLSQR.
	* nslsqr_mf: Same code than nsLSQR but using the approximation in the aux2 module. Mathematically, the code in this module is equivalent to LSQR, so its used a comparison method.
	* qmatrix: Module for the quantized approximation.
	* quant: Module to compress a quantized matrix.

* Python: It contains a Python implementation of the nsLSQR method and the quantized approximation, using numpy and scipy. In particular, its structure is given by:
	* aux_routines.py: Contains the routines to perform a matrix-free product between the Jacobian matrix of a function and an arbitrary vector, and the product between the transpose of the Jacobian matrix and a vector. Recall that the later routine is very slow and inefficient, but is totally matrix-free.
	* nslsqr.py: nsLSQR implementation.
	* qmat.py: Quantized approximation implementation.

## Experiments
The folder 'experiment' contains code to perform the thesis experiment. Inside of each folder there is a README containing an explanation of the experiment and the instruction for execution.

# Contact
If you have some question, comment and you want to contact me, please mail me to ariel.sanhuezar@usm.cl