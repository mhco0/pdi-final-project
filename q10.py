import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import *
from q1 import binarize, two_peaks
from q4 import correlate2d, clip_gray_image
from q5 import edge_dection_kernel
from q8 import dilate

# Image
BOAVIAGEM_BMP_PATH = "./Imagens/Q10/Merge_Timex_BoaViagem.bmp"


def main():
    bv_image = read_images([BOAVIAGEM_BMP_PATH])[0]

    upper_half_bv_image = bv_image[80 : bv_image.shape[0] // 2, :]

    blue_yellow = cv2.cvtColor(upper_half_bv_image, cv2.COLOR_BGR2LAB)[:, :, 2]

    binary_bv_image = binarize(blue_yellow, two_peaks(blue_yellow))

    show_image(binary_bv_image)

    binary_bv_image = np.dstack((binary_bv_image, binary_bv_image, binary_bv_image))

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

    binary_bv_image = cv2.cvtColor(binary_bv_image, cv2.COLOR_BGR2GRAY)

    binary_bv_image = clip_gray_image(
        correlate2d(binary_bv_image, edge_dection_kernel())
    )
    show_image(binary_bv_image)


if __name__ == "__main__":
    main()
