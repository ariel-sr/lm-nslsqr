# Implementation of nsLSQR
import numpy as np
from aux_routines import jac_free, jac_transpose_free
from scipy.linalg import solve_triangular
import sys

# Auxiliar function to compute:
# |  J  | * v
# | s*I |
def augmented_dot(Jv, s, v, m, n):
    M = m+n
    res = np.zeros(M)
    res[:m] = Jv(v)
    res[m:] = s*v
    return res

# Auxiliar function to compute:
# |  J  |^T * w
# | s*I |
def augmented_tdot(JTv, s, w, m, n):
    res = JTv(w[:m])
    res = res + s*w[m:]
    return res

# Implementation of nsLSQR. It solves || F(xi) - J(xi)y || + damp*|| y ||, where J is a quantized jacobian matrix.
# This implementation use nsLSQR to solve the least square problem arising in Levenberg-Marquardt method.
# The quantized matrix is given as parameter, using the implementation of SQQuantizedMatrix, found in sqjac.
# nsLSQR will terminate if rtol or stol is archieved. rtol is the relative residual using the current solution and the initial guess, while
# stol is the slope of a linear regresion of the last lpoints values of the residual, so if stoll is archieved, it will means that
# there are no significative progress towards the solution.
# Needed Parameters:
# - J: Quantized matrix
# - F: Nonlinear function
# - xi: Current solution
# - x0: Initial guess
# Optional parameters
# - Fxi: The vector F(xi). If it is an empty vector, the this value will be computed.
# - rtol (default = 1e-5): Tolerance of the relative residual of the least square function
# - ptol (default = 1e-5): Tolerance for no progress in the current solution
# - ppoints (default = 5): No progress in the current solution of consecutive points
# - lpoints (default = 5): Number of linaer regression points to use to estimate the residual reduction
# - stol (default = 1e-8): Tolerance for the slope of the linear regression of the last lpoints
# - maxit (optional): Max number of iteration in the nsLSQR routine. If None, N will be the max number of iterations
# - maxrestart (default = 1): Number of restarts in the nsLSQR routine
# - damp (default = 1e-5): Regularization parameter for the nsLSQR method
# - apptype (default = quant): Type of method to use. { quant = Quantized and FD, arica = Arica and FD, full-quant = Quantized for both products}
def nslsqr(
# Needed parameters
J, F, xi, x0, 
# Parameters for nsLSQR/LSQR
Fxi=np.array([]), rtol=1e-8, ptol=1e-5, ppoints = 20, lpoints=20, stol=1e-8, maxit=None, maxrestart=1, damp=1e-5, output=False,
# What approximation use
apptype='quant'):

    # Compute some constants vectors
    if len(Fxi) == 0:
        Fxi = F(xi)

    # Matrix dimension
    M, N = J.shape()
    # Problem dimension
    n = N
    m = M + N

    # Big Fxi
    big_Fxi = np.zeros(m)
    big_Fxi[:M] = Fxi.copy()

    # Square of damping factor
    sdamp = np.sqrt(damp)
    
    # Verify max number of iterations
    if maxit == None:
        maxit = n
    maxit = maxit

    # List to store residuals || F(x_k) ||
    residual_list = []
    slope_list = []
    x_list = []

    # Markers for inital residuals
    initial_r0 = -1
    initial_hat_r0 = -1

    # Termination of algorithm flag
    flag = False

    # NO progress counter
    no_progress = 0

    # Restarts iterations
    for restart in range(maxrestart):
        # Use the inital guess for initial residuals.
        # Current problem is || Fxi - J*(x0 + xk) ||
        b = np.zeros(m)
        if apptype == 'full-quant':
            b = augmented_dot(lambda v: J.dot(v), sdamp, x0, M, N)
        else:
            b = augmented_dot(lambda v: jac_free(F, xi, v), sdamp, x0, M, N)
        b = big_Fxi - b

        x_list.append(x0)

        # New problem is || b - J*x_k ||, where x_k = 0 as initial guess

        # Initial solution
        x = np.zeros(n)

        # Initial residuals
        # Least square residual: b
        hat_r0 = b.copy()
        # Normal equation residual: J^T*b
        #r0 = jac_transpose_free(F, xi, b)
        if apptype == 'quant':
            r0 = augmented_tdot(lambda v: J.tdot(v), sdamp, b, M, N)
        elif apptype == 'full-quant':
            r0 = augmented_tdot(lambda v: J.tdot(v), sdamp, b, M, N)
        elif apptype == 'arica':
            r0 = augmented_tdot(lambda w: jac_transpose_free(F, xi, w), sdamp, b, M, N)

        # Norm of both residuals
        hat_r0_norm = np.linalg.norm(hat_r0)
        r0_norm = np.linalg.norm(r0)
        
        # Set the initial residuals
        if initial_r0 < 0:
            initial_r0 = r0_norm
            initial_hat_r0 = hat_r0_norm

        # Store initial residual
        residual_list.append(hat_r0_norm/initial_hat_r0)

        # Build U and V matrix of LSQR, with maxit number of columns
        V = np.zeros((n, maxit+1))
        U = np.zeros((m, maxit+1))

        # Initial vectors u and v
        U[:, 0] = hat_r0/hat_r0_norm
        V[:, 0] = r0/r0_norm

        # B matrix: Used for small least square
        B = np.zeros((maxit+1, maxit))

        # Iterations
        Q = np.zeros((maxit+1, maxit))
        R = np.zeros((maxit, maxit))
        for k in range(maxit):
            # First part of LSQR: Compute next U vector by Gram-Schmidt
            if apptype == 'full-quant':
                y = augmented_dot(lambda v: J.dot(v), sdamp, V[:, k], M, N)
            else:
                y = augmented_dot(lambda v: jac_free(F, xi, v), sdamp, V[:, k], M, N)
            
            for j in range(k+1):
                B[j, k] = np.dot(y, U[:, j])
                y = y - B[j, k]*U[:, j]
            c = np.linalg.norm(y)

            # Test if the solution was found
            if not np.allclose(c, 0):
                # Store the computed U vector
                B[k+1,k] = c
                # If there are iteration left, build next U and V
                if k < n-1:
                    U[:, k+1] = y/c

                    # Flags and perturbation vector for premature break in quantized matrix
                    continue_V_vector = True
                    eps = np.zeros(n)
                    if apptype == 'quant':
                        aux = augmented_tdot(lambda v: J.tdot(v), sdamp, U[:, k+1], M, N)
                    elif apptype == 'full-quant':
                        aux = augmented_tdot(lambda v: J.tdot(v), sdamp, U[:, k+1], M, N)
                    elif apptype == 'arica':
                        aux = augmented_tdot(lambda w: jac_transpose_free(F, xi, w), sdamp, U[:, k+1], M, N)
                    while continue_V_vector:
                        # Second part of LSQR and begin of nsLSQR: Compute next V vector by Gram-Schmidt and quantized matrix
                        y = aux + eps
                        for j in range(k+1):
                            coef = np.dot(y, V[:, j])
                            y = y - coef*V[:, j]
                        c = np.linalg.norm(y)
                        
                        # If c is too small, it means a premature break due to a nonfull-rank quantized matrix J
                        # Solution: add a small perturbation of result
                        if not np.allclose(c, 0):
                            V[:, k+1] = y.copy()/c
                            continue_V_vector = False
                        else:
                            print("Adding a small perturbation in generation of V", file=sys.stderr)
                            eps = np.random.rand(n)
            else:
                flag = True

            # Solve Least-Square problem by the proposed QR decomposition
            # The regularization matriz is assumed to be the identity matrix.
            B_tilde = B[:k+2, :k+1]
            if k == 0:
                y = B_tilde[:, 0].copy()
                R[0, 0] = np.linalg.norm(y)
                Q[:2,0] = y/R[0, 0]
            else:
                #Compute new R
                y = B_tilde[:, -1].copy()
                for j in range(k):
                    R[j, k] = np.dot(y, Q[:(k+1)+1, j])
                    y = y - R[j, k]*Q[:(k+1)+1, j]
                R[k, k] = np.linalg.norm(y)
                Q[:(k+1)+1, k] = y/R[k, k]
            e = hat_r0_norm*Q[0,:k+1]
            c_k = solve_triangular(R[:k+1, :k+1], e)

            # Compute the current solution
            prev_x = x.copy()
            x = np.dot(V[:, :k+1], c_k)
            x_list.append(x0 + x)

            # Measure residual
            if apptype == 'full-quant':
                aux = augmented_dot(lambda v: J.dot(v), sdamp, x, M, N)
            else:
                aux = augmented_dot(lambda v: jac_free(F, xi, v), sdamp, x, M, N)
            residual_list.append(np.linalg.norm(b-aux)/initial_hat_r0)

            # Test of linear regression
            slope = 1
            if len(residual_list) >= lpoints:
                Atol = np.zeros((lpoints, 2))
                btol = np.zeros(lpoints)
                for j in range(lpoints):
                    Atol[j, 1] = 1
                    Atol[j, 0] = j+1
                    btol[j] = residual_list[-1-j]
                slope = np.linalg.lstsq(Atol, btol)[0][0]
                slope_list.append(slope)

            # Test progress solution
            if np.linalg.norm(x - prev_x)/np.linalg.norm(x) <= ptol:
                no_progress += 1
            else:
                no_progress = 0

            # Test for tolerance accomplishment
            if np.abs(slope) <= stol or residual_list[-1] <= rtol or no_progress == ppoints or flag:
                if output:
                    print("Tolerance achieved or solution found")
                return x0+x, residual_list, x_list

        x0 = x0 + x
    return x0, residual_list, x_list
