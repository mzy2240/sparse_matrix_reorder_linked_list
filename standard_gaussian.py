
"""
Solve linear system using LU decomposition and Gaussian elimination
"""

import numpy as np
from scipy.linalg import lu, inv, solve


def gausselim(A,B):
    """
    Solve Ax = B using Gaussian elimination and LU decomposition.

    A = LU   decompose A into lower and upper triangular matrices
    LUx = B  substitute into original equation for A

    Let y = Ux and solve:
    Ly = B --> y = (L^-1)B  solve for y using "forward" substitution
    Ux = y --> x = (U^-1)y  solve for x using "backward" substitution

    :param A: coefficients in Ax = B
    :type A: numpy.ndarray of size (m, n)
    :param B: dependent variable in Ax = B
    :type B: numpy.ndarray of size (m, 1)
    """
    # LU decomposition with pivot
    pl, u = lu(A, permute_l=True)
    # forward substitution to solve for Ly = B
    y = np.zeros(B.size)
    for m, b in enumerate(B.flatten()):
        y[m] = b
        # skip for loop if m == 0
        if m:
            for n in range(m):
                y[m] -= y[n] * pl[m,n]
        y[m] /= pl[m, m]

    # backward substitution to solve for y = Ux
    x = np.zeros(B.size)
    lastidx = B.size - 1  # last index
    for midx in range(B.size):
        m = B.size - 1 - midx  # backwards index
        x[m] = y[m]
        if midx:
            for nidx in range(midx):
                n = B.size - 1 - nidx
                x[m] -= x[n] * u[m, n]
        x[m] /= u[m, m]
    return x.reshape(-1, 1)


if __name__ == '__main__':
    a = np.array([[3, 2], [1, -4]])
    b = np.array([[5], [10]])
    x = gausselim(a, b)
    print(x)
    x = np.array([[5], [10]])/np.array([[3, 2], [1, -4]])
    print(x)
    x = solve(a, b)
    print(x)
    print(np.dot(a, x) == b)
