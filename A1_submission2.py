import cv2
import skimage.util
from skimage import io
import numpy as np
from skimage.color import rgb2gray

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Reading Image
I = cv2.imread('test.jpg', 0)

print(I)

# Pixel Frequency
h,w = I.shape
print(h,w)
hist = np.zeros(256)
for i in np.arange(0,h,1):
  for j in np.arange(0,w,1):
    hist[I[i,j]] += 1

Hist = []
for i in range(0, 256, 4):
  Hist.append(hist[i] + hist[i+1] + hist[i+2] + hist[i+3])
hist = Hist
print(len(hist))

H = np.zeros(64)
for n in np.arange(0,64,1):
  H[n] = H[n-1] + hist[n]

print(len(H))

J = np.zeros((h,w))
for i in np.arange(0,h,1):
  for j in np.arange(0,w,1):
    J[i,j] = np.floor((63/(h*w))*H[I[i,j]//4+1]+0.5)


hist2, bin_edges = np.histogram(J, bins=64, range=(0.0, 64))

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

ax1.imshow(I)
ax1.set_title("Cameraman after histogram equalization")

ax2.plot(hist)
ax2.set_title("Cameraman after histogram equalization")

ax3.imshow(J)
ax3.set_title("Cameraman after histogram equalization")

ax4.plot(hist2)
ax4.set_title("Cameraman after histogram equalization")

plt.show()

