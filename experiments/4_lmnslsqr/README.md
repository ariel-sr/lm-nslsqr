# Summary
This experiment perform the test over lm-nsLSQR. The experiment is performed using CUDA C.

## Instructions
The code expect that the code to be in a folder called 'lmnslsqr'. First, copy the CUDA folder contained in the root of this repo here, and rename it to 'lmnslsqr'. Then, procede to compile the project using the command 'make'. The resulting executable expect the following arguments:

* M: Number of equations in the problem.
* N: Number of unknowns in the problem.
* nlayers: A number indicating how many layers the quantized will have.
* A sequence of nlayers positive integers, indicating the bits used in each layer.
* A number between 0 and 5, including both, that denotes which problem to use. The problem are:
	* 0: Normal problem
	* 1: Uniform problem
	* 2: Sparse I problem
	* 3: Sparse II problem
	* 4: Dense I problem
	* 5: Dense II problem

For example, to solve Dense I problem for 80000 equations and 50000 unknowns, using two layers of 3 bits and one layer of 2 bits is given by:

./exp4 80000 50000 3 3 3 2 4

The folder contains a script called "execute_test.sh" that will execute lm-nsLSQR with different problems, depending on the parameter given to the script. The test will obtain the results of section 7.7 of the Master degree thesis. For example, to execute the test for the Sparse II problem, the script must be executed as follows:

bash execute_test.sh sparse2

The script must be executed after the compilation of the code in the folder. To compile the code, follow the instructiones previously given in this README.
