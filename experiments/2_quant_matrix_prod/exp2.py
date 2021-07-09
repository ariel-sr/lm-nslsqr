import numpy as np
from aux_routines import jac_free, jac_transpose_free
from qmat import QMatrix
import sys

# Compute Quantization memory usage in bytes
def mem(M, N, bit_list):
	L = len(bit_list)
	mem_D = 8*N*L
	mem_Q = M*N*sum(bit_list)/8
	return mem_D + mem_Q

# Read arguments
M = int(sys.argv[1])
N = int(sys.argv[2])
problem = sys.argv[3]
layers = int(sys.argv[4])
bit = int(sys.argv[5])

# load the corresponding problem
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

# SEEDS
seed = 1234
np.random.seed(seed)

# Runs
N_RUNS = 50

# Bits
bits = layers*[bit]

# Random vectors
V = np.random.rand(M, N_RUNS)

# Function
if problem == 'normal' or problem == 'uniform':
    Jv = lambda v: func(v, M)
else:
    F = lambda x: func(M, N, x)
    Jv = lambda v: jac_free(F, xi, v)

# True product
Y = np.zeros((N, N_RUNS))
for k in range(N_RUNS):
    if problem == 'normal' or problem == 'uniform':
        for i in range(N):
            e = np.zeros(N)
            e[i] = 1
            Y[i, k] = np.dot(Jv(e), V[:, k])
    else:
        Y[:, k] = jac_transpose_free(F, xi, V[:, k])

error = np.zeros((len(bits), N_RUNS))
print("Computing for b = {0} bits".format(str(bits)), file=sys.stderr)
J = QMatrix(Jv, M, N, bit_list=bits, norm_tol=-1)
# Iterate over each run
for i in range(N_RUNS):
    for k in range(len(bits)):
        y = J.tdot(V[:, i], nlayers = k+1)
        error[k, i] = np.linalg.norm(Y[:, i] - y)/np.linalg.norm(Y[:, i])
np.save('./output/exp2-{0}-{1}b.npy'.format(problem, bits[0]), error)
