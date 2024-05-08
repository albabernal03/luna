import numpy as np

# Matriz A
A = np.array([
    [10, 7, 1.5],
    [20, 25, 9],
    [30, 5, 42]
])

# Función para realizar la factorización LU sin pivotamiento
def factorizacion_lu(A):
    n = A.shape[0]
    L = np.zeros_like(A, dtype=np.double) # Matriz triangular inferior
    U = np.zeros_like(A, dtype=np.double) # Matriz triangular superior
    for i in range(n):
        L[i, i] = 1
        for j in range(i, n):
            U[i, j] = A[i, j] - L[i, :i] @ U[:i, j]
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - L[j, :i] @ U[:i, i]) / U[i, i]
    return L, U

# Realizamos la factorización LU
L, U = factorizacion_lu(A)

# Suma de las componentes de la matriz U
suma_componentes_U = np.sum(U)

print("Matriz L:\n", L)
print("Matriz U:\n", U)
print("Suma de las componentes de U:", suma_componentes_U)


b = np.array([4.5, 4, 67])

def sustitucion_hacia_adelante(L, b):
    n = L.shape[0]
    y = np.zeros_like(b, dtype=np.double) # Vector solución
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y


y = sustitucion_hacia_adelante(L, b)
suma_componentes_y = np.sum(y)

print("Vector solución y:", y)
print("Suma de las componentes de y:", suma_componentes_y)



det_A = np.prod(np.diag(U))

print("Determinante de A:", det_A)

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
print("Error absoluto:", error_absoluto) # este es el error absoluto entre la solución exacta y la solución obtenida con Jacobi

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

# Aplicamos el método de Gauss-Seidel
x_gs, num_iteraciones = gauss_seidel(A, b, x0)

#mostramos soluciones


print("Solución obtenida con Gauss-Seidel:", x_gs)
print("Número de iteraciones:", num_iteraciones)