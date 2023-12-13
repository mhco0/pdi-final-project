import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import *
from scipy.ndimage import median_filter
from q7 import equalize
from q4 import convolve2d, correlate2d, clip_gray_image
from q8 import erosion, dilate
from q9 import complement_image
from q2 import normalize
from q1 import binarize, two_peaks


# Image
SCENE_BMP_PATH = "./Imagens/Q5/cena.bmp"

# Kernels
LAPLACIAN_FILTER_KERNEL = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
GAUSSIAN_FILTER_KERNEL = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) * (1 / 16)
SOBEL_FILTER_X_KERNEL = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ]
)
SOBEL_FILTER_Y_KERNEL = np.array(
    [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ]
)

# Circles
C1_BMP_PATH = "./Imagens/Q5/C1.bmp"
C2_BMP_PATH = "./Imagens/Q5/C2.bmp"
C3_BMP_PATH = "./Imagens/Q5/C3.bmp"


# Some edge detection kernel
def edge_dection_kernel() -> np.ndarray:
    return convolve2d(LAPLACIAN_FILTER_KERNEL, GAUSSIAN_FILTER_KERNEL)


# Closes the image by dilate then erode
def close_image(image: np.ndarray) -> np.ndarray:
    circle_structure_element = (
        np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype="uint8",
        )
    ) * 255

    dilated_image = dilate(image, circle_structure_element)
    closed_image = erosion(image, circle_structure_element)

    return closed_image.astype("uint8")


# Open some image by eroding then dilating
def open_image(image: np.ndarray) -> np.ndarray:
    circle_structure_element = (
        np.array(
            [
                [1, 1],
                [1, 1],
            ],
            dtype="uint8",
        )
    ) * 255

    erode_image = erosion(image, circle_structure_element)
    opened_image = dilate(erode_image, circle_structure_element)

    return opened_image.astype("uint8")


def main():
    # Read image
    scene_image = read_images([SCENE_BMP_PATH])[0]

    # Assert grey scale
    assert_images_grey_scale(scene_image)

    # Convert to work with only one image
    scene_image = cv2.cvtColor(scene_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian filter
    scene_image = clip_gray_image(correlate2d(scene_image, GAUSSIAN_FILTER_KERNEL))

    show_image(scene_image)

    # Binarize
    binarized_edge_image = clip_gray_image(
        binarize(scene_image, two_peaks(scene_image))
    )

    show_image(binarized_edge_image)

    # Apply Edge detection with Sobel
    sobel_x = correlate2d(binarized_edge_image, SOBEL_FILTER_X_KERNEL)

    sobel_y = correlate2d(binarized_edge_image, SOBEL_FILTER_Y_KERNEL)

    sobel = clip_gray_image(np.abs(sobel_x) + np.abs(sobel_y))

    show_image(sobel)


if __name__ == "__main__":
    main()
