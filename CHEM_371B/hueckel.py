import numpy as np

#This script solves the hückel equations for naphtalene

A = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
              [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
              [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
              [1, 0, 0, 0, 1, 0, 0, 0, 1, 0]])

D, C = np.linalg.eig(A)
sorted_indices = np.argsort(D)
sorted_eigenvalues = D[sorted_indices]
sorted_eigenvectors = C[:, sorted_indices]

print(sorted_eigenvalues)
print(sorted_eigenvectors)

