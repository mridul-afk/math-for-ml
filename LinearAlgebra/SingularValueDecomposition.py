import numpy as np

def SVD(A):
  A = np.array(A,dtype=float)

  U,S,V_t = np.linalg.svd(A,full_matrices=False)
  sigma = np.diag(S)

  return U,sigma,V_t

def reconstruct_matrix(U,sigma,V_t):
  return U @ sigma @V_t

if __name__ == "__main__":
  A = np.array([[3,1],[1,3]],dtype = float)

U ,sigma ,V_t = SVD(A)
A_reconstructed = reconstruct_matrix(U,sigma,V_t)
print("Original Matrix A: \n",A)
print("\nU (Left singular vector): \n",U)
print("\nΣ (Singular values):\n", sigma)
print("\nV^T (Right singular vectors):\n", V_t)
print("\nReconstructed A:\n", A_reconstructed)