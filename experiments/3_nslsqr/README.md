# Summary
This experiment perform the test over nsLSQR. The experiment is performed using CUDA C.

## Instructions
The code expect that the code to be in a folder called 'lmnslsqr'. First, copy the CUDA folder contained in the root of this repo here, and rename it to 'lmnslsqr'. Then, procede to compile the project using the command 'make'. The resulting executable expect the following arguments:

* M: Number of equations in the problem, i.e. the number of rows of the Jacobian matrix.
* N: Number of unknowns in the problem, i.e. the number of columns of the Jacobian matrix.
* nlayers: A number indicating how many layers the quantized will have.
* A sequence of nlayers positive integers, indicating the bits used in each layer.
* A 0 or 1. If 0, then nsLSQR will be executed using a matrix-free routine to compute the transpose of the Jacobian matrix. This is used to mimic LSQR. If the given number is not zero, then nsLSQR will be used with a quantized matrix.
* A number between 0 and 5, including both, that denotes which problem to use. The problem are:
	* 0: Normal problem
	* 1: Uniform problem
	* 2: Sparse I problem
	* 3: Sparse II problem
	* 4: Dense I problem
	* 5: Dense II problem

For example, to execute nsLSQR with the Dense I problem (number 4), with a Jacobian matrix of size 80000x50000 using a total of three layers: two layers of 3 bits and one layer of 2 bits, is given by:

./exp3 80000 50000 3 3 3 2 1 4

where exp3 is the executable produced by the Makefile. If we want to use LSQR, we can use any quantized matrix, since it will not be used. For example, the following execution line will execute the Uniform problem with LSQR:

./exp3 80000 50000 1 2 0 1

The script in execute_test.sh contains an example of how to execute the code, for each test problem. The code contained in the script will execute
 and store the result for the experiment in Section 7.3 of the Master degree thesis. The combination of bits can be modified directly in the script to replicate the results of section 7.4. For example, if we want to execute the experiment for the Dense I problem, the following line should be executed after compiling the code according to the instructions given in this README:

 bash execute_test.sh dense1
 