import numpy as np
import imhist as imhist
from skimage import io

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

image = io.imread("day.jpg")
image = image.ravel()


def get_histogram(image, bins):
    # array with size of bins, set to zeros
    histogram = np.zeros(bins)

    # loop through pixels and sum up counts of pixels
    for pixel in image:
        histogram[int(pixel)] += 1

    # return our final result
    return histogram


hist = get_histogram(image, 256)
plt.plot(hist)
plt.show()

def cumsum(a):
    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    return np.array(b)


# execute the fn
cs = cumsum(hist)

# display the result
# plt.plot(cs)
# plt.show()
