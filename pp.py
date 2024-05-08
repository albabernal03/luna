import numpy as np

# Matriz A
A = np.array([
    [10, 7, 1.5],
    [20, 25, 9],
    [30, 5, 42]
])

def factorizacion_lu(A):
    n = A.shape[0]
    L = np.zeros_like(A, dtype=np.double) 
    U = np.zeros_like(A, dtype=np.double) 
    for i in range(n):
        L[i, i] = 1
        for j in range(i, n):
            U[i, j] = A[i, j] - L[i, :i] @ U[:i, j]
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - L[j, :i] @ U[:i, i]) / U[i, i]
    return L, U

L, U = factorizacion_lu(A)

suma_componentes_U = np.sum(U)


det_A = np.prod(np.diag(U))
print("Determinante de A:", det_A)