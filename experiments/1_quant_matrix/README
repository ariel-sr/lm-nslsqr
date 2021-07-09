# Summary
This experiment perform the comparison between a Jacobian matrix and its quantized approximation, in terms of the quantized error. The experiment is perform in Python and requires numpy and scipy.

## Instructions
The file 'script.sh' is an execution script. It copy the required Python files and then performs the required experiment. It defines three parameters:
* M: The number of "equations" or the number of rows in the Jacobian matrix.
* N: The number of "unknowns" or number of columns in the Jacobian matrix.
* PROBLEM: The problem to test.

By default, the previous parameters are defined for the Dense I problem. The available problems are: 
* dense1
* dense2
* sparse1
* sparse2
* uniform
* normal

Some functions has specific requirements for problem size. Also, as a requirement for the quantized approximation, the value of N must be a multiple of 8. In the thesis experiments, the size for each problem were:
* dense1, dense2, norma, uniform: M = 80000 and N = 50000
* sparse1: M = 155982 and N = 26000
* sparse2: M = N = 63000
