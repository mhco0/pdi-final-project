import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import *

# Images
DOC1_BMP_PATH = "./Imagens/Q1/doc1.bmp"
DOC2_BMP_PATH = "./Imagens/Q1/doc2.bmp"
DOC3_BMP_PATH = "./Imagens/Q1/doc3.bmp"


def binarize(image: np.ndarray, cut_value: int) -> np.ndarray:
    image[np.where(image <= cut_value)] = 0
    image[np.where(image > cut_value)] = 255

    return image


def swap_cut_point(doc1: np.ndarray, doc2: np.ndarray, cut1: int, cut2: int) -> None:
    show_image(binarize(doc1, cut2))
    show_image(binarize(doc2, cut1))


def two_peaks(doc: np.ndarray) -> int:
    histogram, bins = np.histogram(doc.ravel(), 256, [0, 256])
    peak_1 = np.argmax(histogram)

    diffs = np.arange(256).astype("float64")

    for k, h_k in enumerate(histogram):
        diffs[k] = ((k - peak_1) ** 2) * h_k

    peak_2 = np.argmax(diffs)

    return (peak_1 + peak_2) // 2


def my_method_cut_point(image: np.ndarray) -> int:
    histogram, bins = np.histogram(image.ravel(), 256, [0, 256])

    percentile = np.percentile(histogram, 30)

    cut_point = (np.abs(histogram - percentile)).argmin()

    return np.uint8(cut_point)


def main():
    images = read_images([DOC1_BMP_PATH, DOC2_BMP_PATH, DOC3_BMP_PATH])
    images_copy = images.copy()

    assert_images_grey_scale(images)

    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    show_images_hist(images)

    show_image(images[0])
    images[0] = binarize(images[0], 50)
    show_image(images[0])

    show_image(images[1])
    images[1] = binarize(images[1], 70)
    show_image(images[1])

    swap_cut_point(images_copy[0], images_copy[1], 50, 70)

    show_image(images[2])
    images[2] = binarize(images[2], my_method_cut_point(images[2]))
    show_image(images[2])


if __name__ == "__main__":
    main()
