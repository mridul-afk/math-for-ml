import numpy as np

# SINGLE VECTOR PROJECTION
def projection_on_vector(v,u):
   
   # Projecting vector v on vector u
   v = v.astype(float)
   u = u.astype(float)
   return (np.dot(v,u) / np.dot(u,u)) * u

# PROJECTION ON MATRIX

def matrix_projection(p,Q):
   
   # Projecting vector p onto subspace created by the columns of Q assuming Q to be orthonormal
   return Q @ np.transpose(Q) @ p

if __name__ == "__main__":
   v = np.array([2.0,1.0])
   u = np.array([1.0,2.0])
   proj = projection_on_vector(v,u)
   print("Projection of vector v onto vector u:",proj)

   p = np.array([2.0,1.0])

   q1 = np.array([1.0,0.0])
   q2 = np.array([0.0,1.0])
   Q = np.column_stack([q1,q2])

   proj1 = matrix_projection(p,Q)
   print("Projection onto subspace:",proj1)