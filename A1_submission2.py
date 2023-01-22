import cv2
import skimage.util
from skimage import io
import numpy as np
from skimage.color import rgb2gray

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

image = cv2.imread('test.jpg', 0)

hist2, bin_edges = np.histogram(image, bins=64, range=(0, 256))

fig=plt.figure()
plt.plot(hist2)
plt.title("Histogram of Cameraman")
plt.show()