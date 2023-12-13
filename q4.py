import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import *
import scipy.signal as scs


# Image
CAMERAMEN_BMP_PATH = "./Imagens/Q4/cameraman.bmp"

# Filter kernels
BOX_FILTER_KERNEL = np.ones((3, 3)) * (1 / 9)
H1_FILTER_KERNEL = np.ones((1, 3)) * (1 / 3)
H2_FILTER_KERNEL = np.ones((3, 1)) * (1 / 3)


# Applies Correlation between a image and a kernel
def correlate2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return scs.correlate2d(img, kernel, mode="same", boundary="symm")


# Applies Discrete convolution between a image and a kernel
def convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return scs.convolve2d(img, kernel, mode="full")


# Clips content of image to be inside rbg-8 range
def clip_gray_image(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype("uint8")


# Applies a Discrete convolution between the H1 filter kernel and the H2 filter kernel
def convolve_h1_h2() -> np.ndarray:
    return convolve2d(H1_FILTER_KERNEL, H2_FILTER_KERNEL)


def main():
    # Read image
    image = read_images([CAMERAMEN_BMP_PATH])[0]

    # Asserts grey scale
    assert_images_grey_scale([image])

    # Convert just to work with one matrix
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Shows image
    show_image(image)

    # Apply box filter correlation
    image_box_filter = clip_gray_image(correlate2d(image, BOX_FILTER_KERNEL))
    show_image(image_box_filter)

    # Apply two correlations between the image with filter h1 then h2
    image_h1_then_h2 = clip_gray_image(
        correlate2d(correlate2d(image, H1_FILTER_KERNEL), H2_FILTER_KERNEL)
    )

    show_image(image_h1_then_h2)

    # Apply one discrete convolution and one correlation with the image
    image_h1_h2_combined = clip_gray_image(correlate2d(image, convolve_h1_h2()))
    show_image(image_h1_h2_combined)

    # Shows mean error between the two images
    print(np.abs(image_h1_h2_combined - image_box_filter).mean())


if __name__ == "__main__":
    main()
