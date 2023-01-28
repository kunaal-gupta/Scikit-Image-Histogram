import math
import cv2
from skimage import io
import numpy as np

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def BinArrGenerate(n, start, end):
    """ Creating array consisting of n + 1 elements of equal difference from start - end"""
    flag = ((end - start) / (n))

    # Initializing the array with element 0
    BinArr = [start]
    for i in range(n - 1):
        BinArr.append(round((BinArr[-1] + flag), 2))

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


def image_histogram(bin, imageArr):
    # Making array of 64 equal sized bins from 0-256
    bin_arr = BinArrGenerate(bin, 0, 256)

    # Dictionary to store of frequency of pixel falling in a particular bin range
    pixelCountDict = dict()
    for i in bin_arr:
        pixelCountDict[i] = 0

    for pixel in imageArr:
        pixelCountDict[BinCompute(bin_arr, pixel)] += 1

    return pixelCountDict


def part1_histogram_compute():
    """Computing a 64-bin gray scale histogram of the image"""

    # Reading Image in grayscale
    image = cv2.imread('test.jpg', 0)

    # Converting Image array to 1-D array
    fimage = image.ravel()

    # Plotting Self histogram
    plt.subplot(1, 2, 1)
    plt.title("My Histogram")
    plt.plot(list(image_histogram(bin=64, imageArr=fimage).values()))

    # Plotting NumPy histogram
    plt.subplot(1, 2, 2)
    plt.title("Histogram of numpy Image")
    hist2, bin_edges = np.histogram(image, bins=64, range=(0, 256))

    plt.title("Numpy Histogram")

    plt.plot(hist2)
    plt.show()


def part2_histogram_equalization():
    """Performing a 64-bin grayscale histogram equalization"""

    # Reading Image
    I = cv2.imread('test.jpg', 0)

    # Counting Pixel Frequency at each from 0-255
    h, w = I.shape
    hist = np.zeros(256)
    for i in np.arange(0, h, 1):
        for j in np.arange(0, w, 1):
            hist[I[i, j]] += 1

    # Making 64 bin histogram
    Hist = []
    for i in range(0, 256, 4):
        Hist.append(hist[i] + hist[i + 1] + hist[i + 2] + hist[i + 3])
    hist = Hist

    # Calculating Cumulative Histogram
    H = np.zeros(64)
    for n in np.arange(0, 64, 1):
        H[n] = H[n - 1] + hist[n]

    # Applying equalization algorithm
    J = np.zeros((h, w))
    for i in np.arange(0, h, 1):
        for j in np.arange(0, w, 1):
            J[i, j] = np.floor((63 / (h * w)) * H[I[i, j] // 4 + 1] + 0.5)

    # Plotting graphs & images
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    fimage = list(J.ravel())

    # Making array of 64 equal sized bins
    bin_arr = BinArrGenerate(64, 0, 63)

    # Dictionary to store of frequency of pixel falling in a particular bin range
    pixelCountDict = dict()
    for i in bin_arr:
        pixelCountDict[i] = 0

    for pixel in fimage:
        pixelCountDict[BinCompute(bin_arr, pixel)] += 1

    ax1.imshow(I, cmap='gray')
    ax1.set_title("Original Image")

    ax2.plot((image_histogram(bin=64, imageArr=I.ravel())).values())
    ax2.set_title("Histogram")

    ax3.imshow(J, cmap='gray')
    ax3.set_title("New Image")

    ax4.plot(pixelCountDict.values())
    ax4.set_title("Histogram after Equalization")

    plt.show()


def part3_histogram_comparing():
    """ Comparing two images' Histogram to calculate Bhattacharyya Coefficient"""

    # Reading the images
    I1 = cv2.imread('day.jpg', 0)
    I2 = cv2.imread('night.jpg', 0)

    # Pixel Frequency for image 1
    h, w = I1.shape
    hist1 = np.zeros(256)
    for i in np.arange(0, h, 1):
        for j in np.arange(0, w, 1):
            hist1[I1[i, j]] += 1 / (h * w)

    # Pixel Frequency for image 2
    h, w = I2.shape
    hist2 = np.zeros(256)
    for i in np.arange(0, h, 1):
        for j in np.arange(0, w, 1):
            hist2[I2[i, j]] += 1 / (h * w)

    sum = 0

    # Calculating Bhattacharyya Coefficient
    for i in range(256):
        sum += math.sqrt(hist1[i] * hist2[i])
    output = 'Bhattacharyya Coefficient: ' + str(sum)
    print(output)
    return output


def histogram_matching_algorithm(I1, I2):
    """Implementing Histogram matching of two images. It's used to calculate for both colored & grayscale images"""

    # Dimensions of the image
    l, w = I1.shape

    # Histogram
    hist1, bin_edges1 = np.histogram(I1, bins=256, range=(0, 256))
    hist2, bin_edges2 = np.histogram(I2, bins=256, range=(0, 256))

    # New Numpy arrays
    H1 = np.zeros(256, dtype=float)
    H2 = np.zeros(256, dtype=float)

    # Initializing first elements in the new numpy array
    H1[0] = hist1[0]
    H2[0] = hist2[0]

    #  Cumulative Histogram
    for i in range(1, len(hist1)):
        H1[i] = (H1[i - 1] + hist1[i])
        H2[i] = (H2[i - 1] + hist2[i])

    # Normalized Histogram
    for i in range(len(H1)):
        H1[i] /= (l * w)
        H2[i] /= (l * w)

    # Implementing Histogram matching algorithm
    GrayScaleRange = [i for i in range(0, 256)]

    J = np.zeros((l, w))
    A = [0] * 256

    a_ = 0
    for a in GrayScaleRange:
        while H1[a] > H2[a_]:
            a_ += 1
        A[a] = a_

    for i in range(0, l, 1):
        for j in range(0, w, 1):
            a = int(I1[i][j])
            J[i, j] = A[a]

    return J


def histogram_matching_colored():
    """ Histogram Matching for the colored Image"""

    # Reading images in RGB
    I1 = io.imread('day.jpg')
    I2 = io.imread('night.jpg')

    # Breaking image 1 array into RGB arrays
    I1red = I1[:, :, 0]
    I1green = I1[:, :, 1]
    I1blue = I1[:, :, 2]

    # Breaking image 2 array into RGB arrays
    I2red = I2[:, :, 0]
    I2green = I2[:, :, 1]
    I2blue = I2[:, :, 2]

    # New RGB arrys after doing histogram matching of image 1 & image 2
    Jred = histogram_matching_algorithm(I1red, I2red)
    Jgreen = histogram_matching_algorithm(I1green, I2green)
    Jblue = histogram_matching_algorithm(I1blue, I2blue)

    # Joining RGB arrays to form colored image
    rgb = np.dstack((Jred, Jgreen, Jblue))

    return I1, I2, rgb


def part4_histogram_matching():
    """ This function plots final colored & grayscale images & their respective input images """

    # Reading images in grayscale
    I1 = cv2.imread('day.jpg', 0)
    I2 = cv2.imread('night.jpg', 0)

    # Output array of grayscale image after histogram matching
    J = histogram_matching_algorithm(I1, I2)

    # Output array of colored image after histogram matching
    cI1, cI2, Jcol = histogram_matching_colored()

    # Making the fianl images & their input images ready for plotting
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)

    # Grayscale Images
    ax1.imshow(I1, cmap='gray')
    ax1.set_title("source_gs")

    ax2.imshow(I2, cmap='gray')
    ax2.set_title("template_gs")

    ax3.imshow(J, cmap='gray')
    ax3.set_title("matched_gs")

    # RGB Images
    ax4.imshow(cI1)
    ax4.set_title("source_rgb")

    ax5.imshow(cI2)
    ax5.set_title("template_rgb")

    ax6.imshow(Jcol.astype(np.uint16))
    ax6.set_title("matched_rgb")

    plt.show()


if __name__ == '__main__':
    part1_histogram_compute()
    part2_histogram_equalization()
    part3_histogram_comparing()
    part4_histogram_matching()
