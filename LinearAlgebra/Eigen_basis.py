import numpy as np

def diagonalise(T,n):
  T = np.array(T,dtype=float)
  eigen_values , eigen_vectors = np.linalg.eig(T)
  C = eigen_vectors
  D = np.diag(eigen_values ** n)
  C_inv = np.linalg.inv(C)
  T_n = C @ D @ C_inv
  return T_n

if __name__ == "__main__":
  T = np.array([
    [2,1],
    [4,5]
  ],dtype=float)

n = 5
T_n = diagonalise(T,n)

print(f"T^{n}:",T_n)
