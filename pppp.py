import numpy as np
A = np.array([
    [10, 7, 1.5],
    [20, 25, 9],
    [30, 5, 42]
])
b = np.array([4.5, 4, 67])
x0 = np.array([0.0, 0.0, 0.0])

def gauss_seidel(A, b, x0, epsilon=1e-6):
    n = A.shape[0]
    x = x0.copy()
    x_prev = x0.copy()
    diff = np.inf
    iterations = 0
    
    while diff > epsilon:
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * x[j]
            x[i] = (b[i] - sigma) / A[i, i]
        diff = np.linalg.norm(x - x_prev, 2)
        x_prev = x.copy()
        iterations += 1
        
    return x, iterations


x_gs, num_iteraciones = gauss_seidel(A, b, x0)

print("Solución obtenida con Gauss-Seidel:", x_gs)
print("Número de iteraciones:", num_iteraciones)