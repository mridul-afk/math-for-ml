import numpy as np

def inner_product(x,y):
  return np.dot(x,y)

def vector_norm(x):
  return np.sqrt(inner_product(x,x))

def cosine_similarity(x, y):
    return inner_product(x, y) / (vector_norm(x) * vector_norm(y))


x = np.array([2, 3, 1])
y = np.array([1, -1, 2])

ip = inner_product(x, y)
norm_x = vector_norm(x)
norm_y = vector_norm(y)
cos_sim = cosine_similarity(x, y)

print("Inner Product:", ip)
print("Norm of x:", norm_x)
print("Norm of y:", norm_y)
print("Cosine Similarity:", cos_sim)