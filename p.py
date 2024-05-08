import numpy as np

def metodo_potencia(A, v0, N=5):
    v = v0
    for _ in range(N):
        v = np.dot(A, v)
        v = v / np.linalg.norm(v, 2)
    lambda_approx = np.dot(v.T, np.dot(A, v)) / np.dot(v.T, v)
    return lambda_approx

# Matriz A para la sección B
A_B = np.array([
    [2, 5.5, 18],
    [0, 49, 7],
    [0, 0, 50]
])
v0 = np.array([1.0, 1.0, 1.0])


lambda_max = 50


lambda_5 = metodo_potencia(A_B, v0, N=5)
error = abs(lambda_max - lambda_5)

print("Autovalor aproximado en la iteración 5:", lambda_5)
print("Error:", error)

def metodo_potencia_inversa(A, v0, epsilon=1e-4):
    v = v0
    lambda_prev = 0
    iteraciones = 0
    
    while True:
        iteraciones += 1
        # Resolver Ax = v para x, que es equivalente a aplicar A^-1 a v
        x = np.linalg.solve(A, v)
        # Normalizar x para obtener el nuevo v
        v = x / np.linalg.norm(x, 2)
        # Calcular el nuevo autovalor aproximado
        lambda_approx = np.dot(v.T, np.dot(A, v)) / np.dot(v.T, v)
        # Verificar el criterio de parada
        if abs(lambda_approx - lambda_prev) < epsilon:
            break
        lambda_prev = lambda_approx

    return lambda_approx, iteraciones

# Aplicamos el método de la potencia inversa
lambda_min_approx, num_iteraciones = metodo_potencia_inversa(A_B, v0)

#resultados
print("Autovalor mínimo aproximado:", lambda_min_approx)

print("Número de iteraciones:", num_iteraciones)
