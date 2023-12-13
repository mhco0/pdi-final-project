import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import *
from q1 import binarize, two_peaks
from q4 import correlate2d, clip_gray_image
from q5 import edge_dection_kernel, SOBEL_FILTER_X_KERNEL, SOBEL_FILTER_Y_KERNEL
from q8 import dilate

# Image
BOAVIAGEM_BMP_PATH = "./Imagens/Q10/Merge_Timex_BoaViagem.bmp"


def main():
    # Read image
    bv_image = read_images([BOAVIAGEM_BMP_PATH])[0]

    # Get only region of interest
    upper_half_bv_image = bv_image[80 : bv_image.shape[0] // 2, :]
    show_image(upper_half_bv_image)

    # Convert to LAB representation and get only channel with yellow and blue
    blue_yellow = cv2.cvtColor(upper_half_bv_image, cv2.COLOR_BGR2LAB)[:, :, 2]

    # Apply binarization
    binary_bv_image = binarize(blue_yellow, two_peaks(blue_yellow))

    show_image(binary_bv_image)

    # Merge image on grey scale just to work with my dilate algorithm
    binary_bv_image = np.dstack((binary_bv_image, binary_bv_image, binary_bv_image))

    # Applies some dilatation to fill the gaps
    for i in range(7):
        binary_bv_image = clip_gray_image(
            dilate(
                binary_bv_image,
                np.array(
                    [
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                    ],
                    dtype="uint8",
                )
                * 255,
            )
        )

    show_image(binary_bv_image)

    # Convert the image to grey to only work with one matrix again
    binary_bv_image = cv2.cvtColor(binary_bv_image, cv2.COLOR_BGR2GRAY)

    # Applies correlation with the edge dection kernel and finds the ocean line
    sobel_x = correlate2d(binary_bv_image, SOBEL_FILTER_X_KERNEL)
    sobel_y = correlate2d(binary_bv_image, SOBEL_FILTER_Y_KERNEL)

    sobel = clip_gray_image(np.abs(sobel_x) + np.abs(sobel_y))

    show_image(sobel)


if __name__ == "__main__":
    main()
