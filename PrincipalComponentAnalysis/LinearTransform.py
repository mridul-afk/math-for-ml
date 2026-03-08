import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread(r"PrincipalComponentAnalysis\image.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

A = np.array([
    [1.2, 0.3],
    [0.2, 1.0]
])

h, w, _ = img.shape

coords = np.indices((h, w)).reshape(2, -1)
new_coords = A @ coords


new_coords = new_coords - new_coords.min(axis=1, keepdims=True)
new_coords = (new_coords / new_coords.max(axis=1, keepdims=True))

new_h = int(new_coords[0].max() * h)
new_w = int(new_coords[1].max() * w)

transformed = np.zeros((h, w, 3), dtype=np.uint8)

for i in range(coords.shape[1]):
    x, y = coords[:, i]
    nx = int(new_coords[0, i] * (h-1))
    ny = int(new_coords[1, i] * (w-1))
    transformed[nx, ny] = img[x, y]

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(transformed)
plt.title("After Linear Transformation")
plt.axis("off")

plt.show()