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


def part1_histogram_compute():
    # Reading Image in grayscale
    image = io.imread("test.jpg")
    grayImage = rgb2gray(image)
    image = skimage.util.img_as_ubyte(grayImage)

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

    # Plotting Self histogram
    plt.subplot(1, 2, 1)
    plt.title("Self Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixel count")
    plt.plot(list(pixelCountDict.keys()), list(pixelCountDict.values()))

    # Plotting NumPy histogram
    plt.subplot(1, 2, 2)
    plt.title("Histogram of numpy Image")
    hist2, bin_edges = np.histogram(image, bins=64, range=(0, 256))
    plt.title("NumPy Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixel count")

    plt.plot(bin_edges[0:-1], hist2)
    plt.show()


def part2_histogram_equalization():
    """add your code here"""


def part3_histogram_comparing():
    DayImage = skimage.util.img_as_ubyte(rgb2gray(io.imread("day.jpg")))
    NightImage = skimage.util.img_as_ubyte(rgb2gray(io.imread("night.jpg")))

    print(len(DayImage.ravel()))
    print(len(NightImage.ravel()))


    plt.subplot(1, 2, 1)
    plt.title("Histogram of numpy Day Image")
    hist2, bin_edges = np.histogram(DayImage, bins=256, range=(0, 256))
    plt.xlabel("grayscale value")
    plt.ylabel("pixel count")
    plt.plot(bin_edges[0:-1], hist2)

    plt.subplot(1, 2, 2)
    plt.title("Histogram of numpy Night Image")
    hist2, bin_edges = np.histogram(NightImage, bins=256, range=(0, 256))
    plt.xlabel("grayscale value")
    plt.ylabel("pixel count")
    plt.plot(bin_edges[0:-1], hist2)

    plt.show()


def part4_histogram_matching():
    """add your code here"""


if __name__ == '__main__':
    # part1_histogram_compute()
    part2_histogram_equalization()
    part3_histogram_comparing()
    part4_histogram_matching()
