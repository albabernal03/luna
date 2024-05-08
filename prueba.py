#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on Tue Apr  2 14:50:29 2024

@author: lunabernalrodriguez
"""
import numpy as np
from scipy.linalg import lu

# Definir la matriz A
A = np.array([[10., 7., 1.5],
              [20., 25., 9.],
              [30., 5., 42.]])

# Factorización LU
P, L, U = lu(A)

print("Matriz L:")
print(L)

print("\nMatriz U:")
print(U)

# suma de los componentes de U 
suma_componenetes= sum(U) #La verdad qeu no se sumarlo
print(suma_componenetes)

#%% Ejercicio 2
import numpy as np
from scipy.linalg import lu, solve_triangular

A = np.array([[10., 7., 1.5],
              [20., 25., 9.],
              [30., 5., 42.]])

B = np.array([[4.5],
              [4.],
              [67.]])


# Factorización LU de A
P, L, U = lu(A)

Y = solve_triangular(L, np.dot(P, B), lower=True)
# Luego UX = Y
X = solve_triangular(U, Y)

print(X)

suma_componentes = sum(X) # aquí igual 
print(suma_componentes)

#%% Ejercicio 3
#%% Ejercicio 4
import numpy as np

A = np.array([[10., 7., 1.5],
              [20., 25., 9.],
              [30., 5., 42.]])

b = np.array([[4.5],
              [4.],
              [67.]])

x0 = np.zeros_like(b)


def jacobi(A, b, x0, tol):
    n = A.shape[0]
    x = np.copy(x0)
    x_new = np.zeros_like(x)
    converged = False

    while not converged:
        for i in range(n):
            x_new[i] = (b[i] - np.dot(A[i, :], x) + A[i, i]*x[i])/A[i, i]

        if np.linalg.norm(x_new - x)/np.linalg.norm(x) < tol:
            converged = True

        x = np.copy(x_new)
    return x_new


sol1 = jacobi(A, b, x0, 1e-6)
print(sol1)


def solve_iterative_jacobi_LU(A, b, x_0, epsilon=1e-6, max_iterations=10):
    n = len(b)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    D = np.zeros(n)

    # Descomposición LU de la matriz A
    for i in range(n):
        L[i, :i] = A[i, :i]
        U[i, i+1:] = A[i, i+1:]
        D[i] = A[i, i]

    x = x_0.copy()
    x_new = x_0.copy()

    for _ in range(max_iterations):
        x_new = (b - ((L + U) @ x)) / D

        if np.linalg.norm(x_new - x) / np.linalg.norm(x) < epsilon:
            return x_new

        x = x_new.copy()

    return x_new


sol2 = solve_iterative_jacobi_LU(A, b, x0)

print(sol2)

#%% Ejercicio 5
from numpy.linalg import inv
from numpy import array, zeros, matmul, ones
import numpy as np


def make_diagonally_dominant(A, b):
    n = len(b)
    for i in range(n):
        row_sum = np.sum(np.abs(A[i])) - np.abs(A[i, i])
        if np.abs(A[i, i]) <= row_sum:
            A[i, i] = row_sum + 1
    return A


A = np.array([[10., 7., 1.5],
              [20., 25., 9.],
              [30., 5., 42.]])

b = np.array([[4.5],
              [4.],
              [67.]])


def descomponer_LDU(A):
    n = A.shape[0]
    L = zeros((n, n))
    D = zeros((n, n))
    U = zeros((n, n))
    for i in range(n):
        L[i, :i] = A[i, :i]
        D[i, i] = A[i, i]
        U[i, i+1:] = A[i, i+1:]
    return L, D, U


def gauss_seidel(A, b):
    x = ones((A.shape[0]))
    kmax = 100
    L, D, U = descomponer_LDU(A)
    T = -matmul(inv(L + D), U)
    c = matmul(inv(L + D), b)
    for _ in range(kmax):
        x = matmul(T, x) + c
    return x


# Calling the function and printing the result
result = gauss_seidel(A, b)
print("Solution:", result)





from numpy import array, dot, argmax, argmin
from numpy.linalg import norm, eig
from numpy.random import rand

A = array([[2., 5.5, 18.],
           [0., 49., 7.],
           [0., 0., 50.]])


def potencia(A):
    error = 1e-16
    kmax = 100
    u = rand(A.shape[0])
    u = u / norm(u)
    h0 = 0
    for k in range(kmax+1):
        u = dot(A, u)
        u = u / norm(u)
        h = dot(dot(A, u), u)
        if abs(h - h0) < error:
            return h, u
        h0 = h
    return "No hay autovalor dominante"


# Usando el método de la potencia para encontrar el autovalor dominante y su autovector
print("Método de la potencia:")
print(potencia(A))

# Usando la función eig de NumPy para encontrar autovalores y autovectores
autovalores, autovectores = eig(A)

# Encontrar el índice del autovalor dominante
indice_hmax = argmax(abs(autovalores))
hmax = autovalores[indice_hmax]
u_hmax = autovectores[:, indice_hmax]

print("Autovalor dominante y su autovector utilizando eig:")
# eig devuelve el valor absoluto del autovalor y autovector, por lo que se necesita tomar solo uno de ellos
print(hmax, u_hmax)
#%% Ejercicio 7
from numpy import array, dot, argmin
from numpy.linalg import norm, solve, eig
from numpy.random import rand

A = array([[2., 5.5, 18.],
           [0., 49., 7.],
           [0., 0., 50.]])

def potencia_inv(A):
    error = 1e-4
    kmax = 1000
    u = rand(A.shape[0])
    u = u / norm(u)
    h0 = 0
    for k in range(kmax + 1):
        x = solve(A, u)  # Solve the system A * u(k) = u(k-1); u(k) = x
        u = x / norm(x)
        h = dot(dot(A, u), u)
        if abs(h - h0) < error:
            return 1 / h, u
        h0 = h
    return "No hay autovalor dominante"


print(potencia_inv(A))

# Calculate eigenvalues and eigenvectors using eig
autovalores, autovectores = eig(A)
indice_hmin = argmin(abs(autovalores))
hmin = autovalores[indice_hmin]
u_hmin = autovectores[:, indice_hmin]
print(hmin, u_hmin)