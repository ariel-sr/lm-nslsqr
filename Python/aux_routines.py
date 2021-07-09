# This file contains some auxiliary functions used in the LM method implementation

import numpy as np

# Jacobian matrix times vector approximation using a finite difference method.
# Parameters:
# - F: Function whose jacobian is desired
# - x: Evaluation point of the jacobian matrix
# - u: Vector to multiply by the Jacobian matrix
# - order {1, 2}: Method order of the approximation
# - eps: Small step for the approximation
def jac_free(F, x, u, order=2,eps=1e-5):
    x_plus = x + eps*u
    res = F(x_plus)
    if order == 1:
        res = res - F(x)
    elif order == 2:
        x_minus = x - eps*u
        res = res - F(x_minus)
        res = res/2
    return res/eps

# Function to compute J^T * v by Arica proposal
def jac_transpose_free(F, x, u, eps=1e-5, k=None):
    n = len(x)
    m = len(u)
    if k == None:
        k = n
    res = np.zeros(n)
    for i in range(min(k, n)):
        e = np.zeros(n)
        e[i] = 1
        p = jac_free(F, x, e, eps=eps)
        res[i] = np.dot(p, u)
    return res