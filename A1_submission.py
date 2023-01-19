import imhist as imhist
from skimage import io
import numpy as np

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def BinArrGenerate(n):
    """ Creating array consisting of n elements of equal difference from 0-1"""
    flag = round(1 / (n - 1), 3)

    # Initializing the array with element 0
    BinArr = [0]
    for i in range(n - 2):
        BinArr.append(BinArr[-1] + flag)

    # Appending element 1 at the end
    BinArr.append(1)

    return BinArr


def BinCompute(bin_arr, pixel):
    """Computing appropriate bin for a particular pixel"""

    # Iterating binArray in -1 manner to find the bin element which is smaller than pixel
    for i in range(len(bin_arr) - 1, -1, -1):
        if pixel >= bin_arr[i]:

            # Returning the bin element
            return bin_arr[i]


def part1_histogram_compute():
    # Reading Image in grayscale
    image = io.imread("test.jpg", as_gray=True)

    # Image Dimensions
    h, w = image.shape

    # Making array of 64 equal sized bins from 0-1
    bin_arr = BinArrGenerate(64)

    # Dictionary to store of frequency of pixel falling in a particular bin range
    pixelCountDict = dict()
    for i in bin_arr:
        pixelCountDict[i] = 0

    # Calculating pixel frequency
    for i in np.arange(0, h, 1):
        for j in np.arange(0, w, 1):
            pixelCountDict[BinCompute(bin_arr, image[(int(i)), int(j)])] += 1

    # Plotting Self histogram
    plt.subplot(1, 2, 1)
    plt.xlim([0.0, 1.0])  # <- named arguments do not work here
    plt.title("Self Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixel count")
    plt.plot(list(pixelCountDict.keys()), list(pixelCountDict.values()))

    # Plotting NumPy histogram
    plt.subplot(1, 2, 2)
    plt.title("Histogram of numpy Image")
    hist2, bin_edges = np.histogram(image, bins=64, range=(0.0, 1.0))
    plt.title("NumPy Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixel count")
    plt.xlim([0.0, 1.0])  # <- named arguments do not work here

    plt.plot(bin_edges[0:-1], hist2)
    plt.show()


def part2_histogram_equalization():
    """add your code here"""


def part3_histogram_comparing():
    """add your code here"""


def part4_histogram_matching():
    """add your code here"""


if __name__ == '__main__':
    part1_histogram_compute()
    part2_histogram_equalization()
    part3_histogram_comparing()
    part4_histogram_matching()
