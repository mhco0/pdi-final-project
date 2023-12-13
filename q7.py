import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import *
from scipy.ndimage import median_filter
from q1 import two_peaks, binarize
from q2 import normalize, desnormalize
from q4 import convolve2d, correlate2d, clip_gray_image

# Images
RICE_BMP_PATH = "./Imagens/Q7/rice.bmp"

# Kernels
GAUSSIAN_FILTER_KERNEL = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) * (1 / 16)


# Equalize the histogram of the image by dividing the cdf
def equalize(image: np.ndarray) -> np.ndarray:
    bin_counts, bin_edges = np.histogram(image.ravel(), 256, [0, 256])
    bin_counts = np.int32(bin_counts)

    cdf = bin_counts.cumsum()

    cdf_normalized = ((cdf - cdf.min()) * 255) / (cdf.max() - cdf.min())

    equalized_image = cdf_normalized[image]

    return equalized_image.astype("uint8")


def main():
    # Read image
    rice_image = read_images([RICE_BMP_PATH])[0]

    # Assert grey scale
    assert_images_grey_scale(rice_image)

    # Convert to work with only one matrix
    rice_image = cv2.cvtColor(rice_image, cv2.COLOR_RGB2GRAY)

    # Show rice histogram
    show_images_hist([rice_image])

    # Equalize histogram
    rice_image = equalize(rice_image)

    # Show histogram again
    show_images_hist([rice_image])

    # Normalize 
    rice_image, r_max, r_min = normalize(rice_image)

    # Apply power of 2 to get a image with more separated intensities
    rice_image = rice_image**2

    # Desnormalize image
    rice_image = desnormalize(rice_image, r_max, r_min)

    # Show histogram again 
    show_images_hist([rice_image])

    show_image(rice_image)

    # Uses Gaussian filter to reduce noise
    rice_image = clip_gray_image(correlate2d(rice_image, GAUSSIAN_FILTER_KERNEL))

    # Show histogram again 
    show_images_hist([rice_image])
    
    show_image(rice_image)

    # Biniraze image
    show_image(binarize(rice_image, two_peaks(rice_image)))


if __name__ == "__main__":
    main()
