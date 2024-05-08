import numpy as np
A = np.array([
    [10, 7, 1.5],
    [20, 25, 9],
    [30, 5, 42]
])
b = np.array([4.5, 4, 67])

def jacobi(A, b, x0, N=10):
    n = A.shape[0]
    D = np.diag(A)
    R = A - np.diagflat(D)

    x = x0
    for _ in range(N):
        x = (b - np.dot(R, x)) / D
    return x


x0 = np.array([0.0, 0.0, 0.0])

x_exacta = np.array([1.0, -1.0, 1.0])
x_jacobi = jacobi(A, b, x0, N=10)
error_absoluto = np.linalg.norm(x_exacta - x_jacobi, 2)

print("Solución exacta:", x_exacta)
print("Solución obtenida con Jacobi:", x_jacobi)
print("Error absoluto:", error_absoluto) 