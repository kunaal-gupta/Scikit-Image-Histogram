# Image Processing with Histogram Techniques

## Overview

This project implements various image processing techniques, focusing on grayscale and color image histogram operations. The code uses Python and popular libraries like OpenCV, Matplotlib, NumPy, and Scikit-image to compute histograms, perform histogram equalization, and apply histogram matching for both grayscale and colored images.

## Features

- **Histogram Computation:** Calculate a 64-bin grayscale histogram for an image.
- **Histogram Equalization:** Perform histogram equalization to enhance contrast in grayscale images.
- **Histogram Comparison:** Compare histograms of two images using the Bhattacharyya Coefficient, a similarity measure.
- **Histogram Matching:** Match histograms between two grayscale or colored images.

## Prerequisites

To run this project, you'll need Python 3.x installed along with the following Python libraries:

- `opencv-python`
- `scikit-image`
- `numpy`
- `matplotlib`

You can install the required dependencies by running the following command:

```bash
pip install opencv-python scikit-image numpy matplotlib
```

## Usage
## Part 1: Histogram Computation
Compute a 64-bin grayscale histogram of an image (test.jpg) and compare it with the histogram computed using NumPy's built-in function.

## Part 2: Histogram Equalization
Apply histogram equalization to a grayscale image (test.jpg), enhancing its contrast by redistributing pixel intensities.

## Part 3: Histogram Comparison
Compare the histograms of two grayscale images (day.jpg and night.jpg) by calculating the Bhattacharyya Coefficient to measure the similarity between them.

## Part 4: Histogram Matching
Match histograms between two grayscale images (day.jpg and night.jpg) and between two colored images by handling each RGB channel independently.

## Running the Program
Run the main script to execute all the functions:

## Functions
- `BinArrGenerate(n, start, end)`: Generates an array consisting of n + 1 elements with equal differences from start to end.
- `BinCompute(bin_arr, pixel)`: Computes the appropriate bin for a given pixel based on the bin array.
- `image_histogram(bin, imageArr)`: Calculates the histogram of an image using the specified number of bins.
- `part1_histogram_compute()`: Calculates and plots the 64-bin grayscale histogram of an image using both custom logic and NumPy's built-in histogram function.
- `part2_histogram_equalization()`: Performs histogram equalization on a grayscale image to enhance its contrast.
- `part3_histogram_comparing()`: Compares histograms of two images using the Bhattacharyya Coefficient and outputs the result.
- `histogram_matching_algorithm(I1, I2)`: Implements histogram matching between two grayscale images, adjusting the pixel values of the first image to match the histogram of the second.
- `histogram_matching_colored()`: Implements histogram matching for colored images by applying histogram matching independently to the red, green, and blue channels.
- `part4_histogram_matching()`: Displays the results of histogram matching for both grayscale and colored images.
