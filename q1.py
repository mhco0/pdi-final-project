import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import *

# Images
DOC1_BMP_PATH = "./Imagens/Q1/doc1.bmp"
DOC2_BMP_PATH = "./Imagens/Q1/doc2.bmp"
DOC3_BMP_PATH = "./Imagens/Q1/doc3.bmp"


# Binarize image using @p cut_value and threshold
def binarize(image: np.ndarray, cut_value: int) -> np.ndarray:
    image[np.where(image <= cut_value)] = 0
    image[np.where(image > cut_value)] = 255

    return image


# Swaps the ideal cut point @p cut1 with @p doc2 and uses ideal cut @p cut2 in @p doc1
def swap_cut_point(doc1: np.ndarray, doc2: np.ndarray, cut1: int, cut2: int) -> None:
    show_image(binarize(doc1, cut2))
    show_image(binarize(doc2, cut1))


# Gets the cut_point of a image based on the 30 percentile of the histogram distribution
def my_method_cut_point(image: np.ndarray) -> int:
    histogram, bins = np.histogram(image.ravel(), 256, [0, 256])

    percentile = np.percentile(histogram, 30)

    cut_point = (np.abs(histogram - percentile)).argmin()

    return np.uint8(cut_point)


def main():
    # Load images
    images = read_images([DOC1_BMP_PATH, DOC2_BMP_PATH, DOC3_BMP_PATH])
    images_copy = images.copy()

    # Assert if image is grey scale
    assert_images_grey_scale(images)

    # Properly covert to gray scale to only work with one matrix
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Show histogram of all images
    show_images_hist(images)

    # Gets manual cut point from doc 1 and show images to user
    show_image(images[0])
    images[0] = binarize(images[0], 50)
    show_image(images[0])

    # Gets manual cut point from doc 2 and show images to user
    show_image(images[1])
    images[1] = binarize(images[1], 70)
    show_image(images[1])

    # Swaps cut points with doc 1 and doc 2 and show images to user
    swap_cut_point(images_copy[0], images_copy[1], 50, 70)

    # Apply my function on doc 3 and show images to user
    show_image(images[2])
    cut_point = my_method_cut_point(images[2])
    images[2] = binarize(images[2], cut_point)
    show_image(images[2])

    # Console log cut point
    print(cut_point)


if __name__ == "__main__":
    main()
