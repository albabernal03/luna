import numpy as np
A_B = np.array([
    [2, 5.5, 18],
    [0, 49, 7],
    [0, 0, 50]
])
v0 = np.array([1.0, 1.0, 1.0])

def metodo_potencia_inversa(A, v0, epsilon=1e-4):
    v = v0
    lambda_prev = 0
    iteraciones = 0
    
    while True:
        iteraciones += 1
        x = np.linalg.solve(A, v)
        v = x / np.linalg.norm(x, 2)
        lambda_approx = np.dot(v.T, np.dot(A, v)) / np.dot(v.T, v)
        if abs(lambda_approx - lambda_prev) < epsilon:
            break
        lambda_prev = lambda_approx

    return lambda_approx, iteraciones

lambda_min_approx, num_iteraciones = metodo_potencia_inversa(A_B, v0)

print("Autovalor mínimo aproximado:", lambda_min_approx)
print("Número de iteraciones:", num_iteraciones)
