import matplotlib.pyplot as plt
import numpy as np

import cv2



image = cv2.imread("luna/7.png", cv2.IMREAD_GRAYSCALE).astype(np.float64)
# image = cv2.imread("results/dpt-hybrid-midas-stack/union_results_depth_[1024, 1024].png", cv2.IMREAD_GRAYSCALE).astype(np.float64)
# image = cv2.imread("DPT_base.png", cv2.IMREAD_GRAYSCALE).astype(np.float64)



import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from matplotlib.ticker import LinearLocator

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(20, 20))

# Make data.
X = np.arange(0, image.shape[1], 1)
Y = np.arange(0, image.shape[0], 1)
X, Y = np.meshgrid(X, Y)


kernel_size = 40

kernel = np.ones((kernel_size, kernel_size), np.float32) / kernel_size**2

image = cv2.filter2D(image, -1, kernel)

image = cv2.filter2D(image, -1, kernel)

image = image.astype(np.float64)
# a - ((np.max(a) - np.min(a)) * 0.2 + np.min(a)) 

for index in range(image.shape[0]):
    a = image[index, :]
    a = a - ((np.max(a) - np.min(a)) * 0.2 + np.min(a)) 
    image[index, :] = a


for index in range(image.shape[0]):
    a = image[index, :]
    a = a  - np.min(a)
    image[index, :] = a



# image = np.where(image < 30, 0, 255) 



image = image - np.min(image)

image = image / np.max(image)
image = image * 255


Z = image



# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)


cv2.imwrite("test_display.png", image)

# Customize the z axis.
ax.set_zlim(0, 255)
ax.zaxis.set_major_locator(LinearLocator(10))

ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()