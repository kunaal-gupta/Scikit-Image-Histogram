import math

import cv2
import skimage
import skimage.util
from skimage import io
import numpy as np
from skimage.color import rgb2gray

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def BinArrGenerate(n, start, end):
    """ Creating array consisting of n elements of equal difference from start - end"""
    flag = int((end - start) / (n - 1))

    # Initializing the array with element 0
    BinArr = [start]
    for i in range(n - 2):
        BinArr.append(BinArr[-1] + flag)

    # Appending element 'end' at the end
    BinArr.append(end)

    return BinArr


def BinCompute(bin_arr, pixel):
    """Computing appropriate bin for a particular pixel"""

    # Iterating binArray in -1 manner to find the bin element which is smaller than pixel
    for i in range(len(bin_arr) - 1, -1, -1):
        if pixel >= bin_arr[i]:
            # Returning the bin element
            return bin_arr[i]


def CummulativeHistogram(Hist):
    prev = 0
    for i in Hist:
        Hist[i] = Hist[i] + prev
        prev = Hist[i]

    return Hist

def part1_histogram_compute():
    # Reading Image in grayscale
    image = cv2.imread('test.jpg', 0)
    # print(image)

    # image = io.imread("test.jpg")
    # grayImage = rgb2gray(image)
    # image = skimage.util.img_as_ubyte(grayImage)

    # Converting Image array to 1-D array
    fimage = image.ravel()

    # Making array of 64 equal sized bins from 0-256
    bin_arr = BinArrGenerate(64, 0, 256)

    # Dictionary to store of frequency of pixel falling in a particular bin range
    pixelCountDict = dict()
    for i in bin_arr:
        pixelCountDict[i] = 0

    for pixel in fimage:
        pixelCountDict[BinCompute(bin_arr, pixel)] += 1

    print(pixelCountDict.keys())
    print()

    # Plotting Self histogram
    plt.subplot(1, 2, 1)
    plt.title("Self Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixel count")
    plt.plot(list(pixelCountDict.values()))

    # Plotting NumPy histogram
    plt.subplot(1, 2, 2)
    plt.title("Histogram of numpy Image")
    hist2, bin_edges = np.histogram(image, bins=64, range=(0, 256))
    print((hist2))
    print(bin_edges[0:])
    plt.title("NumPy Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixel count")

    plt.plot(hist2)
    plt.show()
def part2_histogram_equalization():
    # Reading Image
    I = cv2.imread('test.jpg', 0)

    # Pixel Frequency
    h, w = I.shape
    print(h, w)
    hist = np.zeros(256)
    for i in np.arange(0, h, 1):
        for j in np.arange(0, w, 1):
            hist[I[i, j]] += 1

    Hist = []
    for i in range(0, 256, 4):
        Hist.append(hist[i] + hist[i + 1] + hist[i + 2] + hist[i + 3])
    hist = Hist
    print(len(hist))

    H = np.zeros(64)
    for n in np.arange(0, 64, 1):
        H[n] = H[n - 1] + hist[n]

    print(len(H))

    J = np.zeros((h, w))
    for i in np.arange(0, h, 1):
        for j in np.arange(0, w, 1):
            J[i, j] = np.floor((63 / (h * w)) * H[I[i, j] // 4 + 1] + 0.5)

    hist2, bin_edges = np.histogram(J, bins=64)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    ax1.imshow(I, cmap='gray')
    ax1.set_title("Original Image")

    ax2.plot(hist)
    ax2.set_title("Histogram")

    ax3.imshow(J, cmap='gray')
    ax3.set_title("New Image")

    ax4.plot(hist2)
    ax4.set_title("Histogram after Equalization")

    plt.show()
def part3_histogram_comparing():
    I1 = cv2.imread('day.jpg', 0)
    I2 = cv2.imread('night.jpg', 0)

    # Pixel Frequency
    h, w = I1.shape
    hist1 = np.zeros(256)
    for i in np.arange(0, h, 1):
        for j in np.arange(0, w, 1):
            hist1[I1[i, j]] += 1/(h*w)

    h, w = I2.shape
    hist2 = np.zeros(256)
    for i in np.arange(0, h, 1):
        for j in np.arange(0, w, 1):
            hist2[I2[i, j]] += 1/(h*w)

    sum = 0
    for i in range(256):
        sum += math.sqrt(hist1[i]*hist2[i])
    print(sum)

def part4_histogram_matching():
    """add your code here"""


if __name__ == '__main__':
    # part1_histogram_compute()
    # part2_histogram_equalization()
    part3_histogram_comparing()
    # part4_histogram_matching()
