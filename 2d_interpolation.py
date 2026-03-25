import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2



image = cv2.imread("results/dpt-hybrid-midas-stack/union_results_depth_[64, 64].png", cv2.IMREAD_GRAYSCALE)

print(image.shape)


XI, YI = np.meshgrid(np.linspace(0, 255, image.shape[1]), np.linspace(0, 255, image.shape[0]))

ZI = image

print(ZI.shape, XI.shape, YI.shape)


# plot the result
n = plt.Normalize(0, 255)
plt.subplot(1, 1, 1)
plt.pcolor(XI, YI, ZI, cmap=cm.jet)
plt.title('RBF interpolation - multiquadrics')
plt.xlim(0, 255)
plt.ylim(0, 255)
plt.colorbar()
plt.show()