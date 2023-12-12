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


def edge_dection_kernel() -> np.ndarray:
    return convolve2d(LAPLACIAN_FILTER_KERNEL, GAUSSIAN_FILTER_KERNEL)


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
    custom_contour_extraction(SCENE_BMP_PATH)

    scene_image = read_images([SCENE_BMP_PATH])[0]

    assert_images_grey_scale(scene_image)

    scene_image = equalize(scene_image)

    binarized_edge_image = clip_gray_image(
        binarize(scene_image, two_peaks(scene_image))
    )

    binarized_edge_image = cv2.cvtColor(binarized_edge_image, cv2.COLOR_BGR2GRAY)

    sobel_x = correlate2d(binarized_edge_image, SOBEL_FILTER_X_KERNEL)

    sobel_y = correlate2d(binarized_edge_image, SOBEL_FILTER_Y_KERNEL)

    sobel = np.abs(sobel_x) + np.abs(sobel_y)

    degrees = np.arctan(np.abs(sobel_y) / np.abs(sobel_x))

    show_image(degrees)

    norm, ma, mi = normalize(sobel)

    show_images_hist([norm])

    # show_image(norm)

    # show_image(clip_gray_image(correlate2d(t, LAPLACIAN_FILTER_KERNEL)))

    # scene_image = binarize(scene_image, two_peaks(scene_image))

    # closed_scene_image = close_image(scene_image)

    # closed_scene_image = cv2.cvtColor(closed_scene_image, cv2.COLOR_BGR2GRAY)

    # edge_image = correlate2d(closed_scene_image, edge_dection_kernel())

    # binarized_edge_image = clip_gray_image(binarize(edge_image, two_peaks(edge_image)))

    # binarized_edge_image = cv2.cvtColor(binarized_edge_image, cv2.COLOR_GRAY2BGR)

    # show_image(binarized_edge_image)


if __name__ == "__main__":
    main()
