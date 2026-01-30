import numpy as np

def gram_schmidt(basis_vectors):
  Q=[]

  for v in basis_vectors:
    u=v.copy()

    for q in Q:
      u -= np.dot(u,q) * q


    q_new = u / np.linalg.norm(u)
    Q.append(q_new)

  return Q
  

if __name__ == "__main__":
  v1 = np.array([1.0, 1.0])
  v2 = np.array([1.0, 0.0])

  G = gram_schmidt([v1,v2])
  print(G)