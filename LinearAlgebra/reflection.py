import numpy as np

def vector_projection(v,u):
  return (np.dot(v,u) / np.dot(u,u)) * u

def reflect_across_vector(v,u):
  # Reflect vector v across the line made by vector u
  proj = vector_projection(v,u)
  return 2 * proj - v

def gram_schmidt(vectors):
  Q=[]

  for v in vectors:
    u=v.copy()

    for q in Q:
      u -= np.dot(u,q) * q
    q_new = u / np.linalg.norm(u)
    Q.append(q_new)
  return Q


def mirror_basis(v1,v2):
  n = np.cross(v1,v2)
  n = n / np.linalg.norm(n)

  e1,e2,e3 = gram_schmidt([v1,v2,n])
  return np.column_stack([e1,e2,e3])


def reflection(r,v1,v2):
  E = mirror_basis(v1,v2)
  T_E = np.diag([1,1,-1])
  r_reflected = E @ T_E @ E.transpose() @ r
  return r_reflected


if __name__ == "__main__":

    r = np.array([2.0, 3.0, 4.0])

  
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])

    r_reflected = reflection(r, v1, v2)

    print("Original vector r:")
    print(r)

    print("\nReflected vector r':")
    print(r_reflected)
