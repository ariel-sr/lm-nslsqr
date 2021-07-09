# Scalar Quantized Matrix implementation
# This file contains an improved implementation of Scalar Quantization matrix based on columns.
# This implementations avoid the excesive computation of previous implementation by computing all levels of quantization at once.

import numpy as np

class QMatrix:
    # Construction
    # Parameters:
    #   - func: Compute matrix-vector product with the matrix to quantize
    #   - m, n: Problem size
    #   - bit_list: Number of bits to use in each level
    #   - norm_tol: A stopping tolerance for Frobenious norm
    def __init__(self, func, m, n, bit_list=[4, 4], norm_tol=1e-5):
        # Problem size
        self.m = m
        self.n = n
        # Number of packs
        self.p = n//8
        # Number of levels
        self.n_levels = 0
        # List of diagonal scaling matrices
        self.D_list = []
        # Frobenius norm of the quantized matrix, the original matrix and the approximation
        self.qmat_norm  = 0
        self.mat_norm   = 0
        self.error_norm = 0
        # Bit list
        self.bit_list = []
        # Shift list
        self.S_list = []
        # List of compression matrices
        self.M_list = []

        # Fill initial data of hypothetic number of levels
        self.n_levels = len(bit_list)
        for i in range(self.n_levels):
            self.D_list.append(np.zeros(n))
            self.bit_list.append(bit_list[i])
            self.S_list.append( 2**(bit_list[i]-1)-1 )
            self.M_list.append( np.zeros((self.m, self.p, self.bit_list[i]), dtype=np.uint8) )

        # List of norms to track progress
        qmat_norm_list = self.n_levels*[0]
        self.error_norm_list = self.n_levels*[0]
        # Canonical vector
        e = np.zeros(n)
        # Counter for number of columns, current pack and a matrix to compress
        count_cols = 0
        current_pack = 0
        data_to_compress = []
        for i in range(self.n_levels):
            data_to_compress.append(np.zeros((self.m, 8, self.bit_list[i]), dtype=np.uint8))

        # Iteration to quantize each column
        for k in range(n):
            # Get the k-th column
            e[k] = 1
            aux = func(e)
            e[k] = 0
            # Sum vector for norm computation
            sum_aux = np.zeros(m)

            # Increase columns processed
            count_cols += 1
            # Compute norm of original matrix
            self.mat_norm += np.linalg.norm(aux)**2

            # Compute vectors for each level
            for i in range(self.n_levels):
                current_aux = np.zeros(m, dtype=np.uint8)
                # Get max value
                P = np.max(np.abs(aux))
                # Discard small values
                if not np.allclose(P, 0):
                    coef = P/self.S_list[i]
                    current_aux = np.rint(aux/coef + self.S_list[i])
                    self.D_list[i][k] = coef
                # Compute norms
                approx_aux = self.D_list[i][k]*(current_aux - self.S_list[i])
                sum_aux = sum_aux + approx_aux
                qmat_norm_list[i] = qmat_norm_list[i] + np.linalg.norm(sum_aux)**2
                error_col = aux - approx_aux
                self.error_norm_list[i] = self.error_norm_list[i] + np.linalg.norm(error_col)**2
                # Compress the data
                current_aux = current_aux.reshape((current_aux.shape[0], 1)).astype(np.uint8)
                # Get the binary representation
                M = np.unpackbits(current_aux, axis=1)[:, -self.bit_list[i]:]
                data_to_compress[i][:, count_cols-1, :] = M
                # The error is the new aux to quantize
                aux = error_col.copy()

            # Compres if we have already 8 columns
            if count_cols == 8:
                count_cols = 0
                for i in range(self.n_levels):
                    self.M_list[i][:, current_pack, :] = np.packbits(data_to_compress[i], axis=1)[:,0,:]
                current_pack += 1

        # Evaluate the achieved norm
        self.mat_norm = np.sqrt(self.mat_norm)
        idx = self.n_levels
        for i in range(self.n_levels):
            if np.sqrt(self.error_norm_list[i])/self.mat_norm < norm_tol:
                idx = i+1
                break
        # Get the number of element that achieves the norm
        if idx != self.n_levels:
            self.n_levels = idx
            self.D_list = self.D_list[:idx]
            self.bit_list = self.bit_list[:idx]
            self.S_list = self.S_list[:idx]
            self.M_list = self.M_list[:idx]
        self.error_norm_list = np.sqrt(np.array(self.error_norm_list))
        self.error_norm = self.error_norm_list[idx-1]
        self.qmat_norm = np.sqrt(qmat_norm_list[idx-1])

    # Computes J^T * v. Vector v size is m
    def tdot(self, vector, nlayers = None):
        y = np.zeros(self.n)
        s = np.sum(vector)
        if nlayers == None:
            nlayers = self.n_levels
        for i in range(nlayers):
            aux = np.zeros(self.n)
            for k in range(self.p):
                H = np.unpackbits(self.M_list[i][:, k, :], axis=1).T
                r = np.dot(H, vector)
                for j in range(self.bit_list[i]):
                    aux[8*k:8*(k+1)] = aux[8*k:8*(k+1)] + 2**(self.bit_list[i] - 1 - j) * r[8*j:8*(j+1)]
            aux = aux - s*self.S_list[i]
            aux = ( self.D_list[i] * aux)
            y = y + aux
        return y

    # Computes J*v. Vector v size is n
    def dot(self, vector, nlayers = None):
        y = np.zeros(self.m)
        if nlayers == None:
            nlayers = self.n_levels
        for i in range(nlayers):
            Dv = self.D_list[i]*vector
            s = np.sum(Dv)
            aux = np.zeros(self.m)
            power = np.flip(np.repeat(2**np.arange(self.bit_list[i]), 8))
            for k in range(self.p):
                H = np.unpackbits(self.M_list[i][:, k, :], axis=1)
                H = H*power
                mini_v = np.tile(Dv[8*k:8*(k+1)], self.bit_list[i])
                aux = aux + np.dot(H, mini_v)
            aux = aux - self.S_list[i]*s
            y = y + aux
        return y

    def shape(self):
        return (self.m, self.n)

    ############################################################
    ########### DEBUG FUNCTIONS ################################
    ############################################################

    # Get the jacobian matrix
    def get_jacobian(self):
        T = np.zeros((self.m, self.n))
        T_list = []
        for i in range(self.n_levels):
            Th = np.zeros((self.m, self.n))
            for k in range(self.bit_list[i]):
                Th = Th + 2**(self.bit_list[i] - 1 - k) * np.unpackbits(self.M_list[i][:, :, k], axis=1)
            T_list.append(np.dot(Th - self.S_list[i], np.diag(self.D_list[i])))
            T = T + T_list[-1]
        return T, T_list


