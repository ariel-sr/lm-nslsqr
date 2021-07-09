# Set un functions to be used in the thesis

import numpy as np
import sys

################################################################################

# Dense I function
# Based on trigonometric function
def dense1(M, N, x):
	if M < N:
		print("Wrong dimensions", file=sys.stderr)
		sys.exit(1)

	f = np.zeros(M)
	
	# Compute number of chunks
	P = M//N
	I = M - P*N
	vals = []
	for p in range(P):
		coef = np.sum( np.cos(x)**(p+1) )
		vals.append(coef)
	if I == 0:
		for p in range(P+1):
			f[p*N:(p+1)*N] = N - vals[p] + np.arange(p*N+1, (p+1)*N+1)*(1 - np.cos(x)) - np.sin(x)
	else:
		for p in range(P):
			f[p*N:(p+1)*N] = N - vals[p] + np.arange(p*N+1, (p+1)*N+1)*(1 - np.cos(x)) - np.sin(x)
		f[P*N:] = N - vals[-1] + np.arange(P*N+1, M+1)*(1 - np.cos(x[:I])) - np.sin(x[:I])

	return f

# Dense II
# Logarithmic function
def dense2(M, N, x):
	if M < N:
		print("Wrong dimensions", file=sys.stderr)
		sys.exit(1)

	f = np.zeros(M)
	
	# Compute number of chunks
	P = M//N
	I = M - P*N
	s = np.log(1 + np.sum(x**2))
	if I == 0:
		for p in range(P+1):
			f[p*N:(p+1)*N] = x**(p+1)*s + x
	else:
		for p in range(P):
			f[p*N:(p+1)*N] = x**(p+1)*s + x
		f[P*N:] = x[:I]**P * s + x[:I]

	return f

################################################################################

# Normal function N(0, 1)
def normalfunc(e, M, m=0, std=1, scal=2):
    idx = 3*scal*np.where(e == 1)[0][0]
    np.random.seed(idx)
    return np.random.normal(m, std, M)

# Uniform function in [-1, 1]
def uniformfunc(e, M, scal=2):
    idx = 3*scal*np.where(e == 1)[0][0]
    np.random.seed(idx)
    return 2*np.random.rand(M) - 1

################################################################################

# Sparse I
# Quadratic function, by Toint.
def sparse1(M, N, x):
	if M != 6*(N-3):
		print("Wrong dimensions", file=sys.stderr)
		sys.exit(1)

	f = np.zeros(M)
	
	f[::6]  = x[:-3] + 3*x[1:-2]*(x[2:-1] - 1) + x[3:]**2 - 1
	f[1::6] = (x[:-3] + x[1:-2])**2 + (x[2:-1] - 1)**2 - x[3:] - 3
	f[2::6] = x[:-3]*x[1:-2] - x[2:-1]*x[3:]
	f[3::6] = 2*x[:-3]*x[2:-1] + x[1:-2]*x[3:] - 3
	f[4::6] = (x[:-3] + x[1:-2] + x[2:-1] + x[3:])**2 + (x[:-3] - 1)**2
	f[5::6] = x[:-3]*x[1:-2]*x[2:-1]*x[3:] + (x[3:] - 1)**2 - 1

	return f

# Sparse II
# Diagonal function premultiplied by a quasi-orthonormal matrix
def sparse2(M, N, x):
	if M != N or N % 3 != 0:
		print("Wrong dimensions", file=sys.stderr)
		sys.exit(1)

	i = np.arange(N/3)+1
	f = np.zeros(M)
	f[::3]  = 0.6*x[::3] + 1.6*x[::3]**3 - 7.2*x[1::3]*+2 + 9.6*x[1::3] - 4.8
	f[1::3] = 0.48*x[::3] - 0.72*x[1::3]**3 + 3.24*x[1::3]**2 - 4.32*x[1::3] - x[2::3] + 0.2*x[2::3]**3 + 2.16
	f[2::3] = 1.25*x[2::3] - 0.25*x[2::3]**3
	return f