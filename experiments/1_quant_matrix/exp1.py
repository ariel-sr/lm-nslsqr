import numpy as np
from qmat import QMatrix
from aux_routines import jac_free
import sys

# Get parameters from command line. The format is
# python exp1.py M N problem

# Import functions
M = int(sys.argv[1])
N = int(sys.argv[2])
problem = sys.argv[3]

if problem == "dense1":
    from func import dense1 as func
    xi = 2*np.random.rand(N)-1
elif problem == "dense2":
    from func import dense2 as func
    xi = 2*np.random.rand(N)-1
elif problem == "sparse1":
    from func import sparse1 as func
    xi = 2*np.random.rand(N)-1
elif problem == "sparse2":
    from func import sparse2 as func
    xi = 2*np.random.rand(N)-1
elif problem == "normal":
    from func import normalfunc as func
    xi = 2*np.random.rand(N)-1
elif problem == "uniform":
    from func import uniformfunc as func
    xi = 2*np.random.rand(N)-1
else:
    print("Function not found")

# Compute Quantization memory usage in bytes
def mem(M, N, bit_list):
	L = len(bit_list)
	mem_D = 8*N*L
	mem_Q = M*N*sum(bit_list)/8
	return mem_D + mem_Q

# SEEDS
seed = 1234
np.random.seed(seed)

# Bits
bits = [2,4,6,8]
n_layers = 4

# Runs
n_runs = 25

# Function
if problem == 'normal' or problem == 'uniform':
    Jv = lambda v: func(v, M)
    error = np.zeros((len(bits), n_runs, n_layers))
else:
    F = lambda x: func(M, N, x)
    Jv = lambda v: jac_free(F, xi, v)
    error = np.zeros((len(bits), n_layers))

for k in range(len(bits)):
    b = bits[k]
    if problem == 'normal' or problem == 'uniform':
        # Iterate over each run
        for i in range(n_runs):
            # Function
            Jv = lambda v: func(v, M, scal=i+1)
            print("Computing for {0} bits, run {1}".format(b, i+1), file=sys.stderr)
            J = QMatrix(Jv, M, N, bit_list=n_layers*[b], norm_tol=1e-12)
            error[k, i, :] = J.error_norm_list/J.mat_norm
    else:
        print("Computing for b = {0} bits".format(b), file=sys.stderr)
        J = QMatrix(Jv, M, N, bit_list=n_layers*[b], norm_tol=1e-12)
        error[k] = J.error_norm_list/J.mat_norm
np.save('./output/exp1-{0}.npy'.format(problem), error)
