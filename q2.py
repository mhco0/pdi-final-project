import numpy as np
import cv2
import matplotlib.pyplot as plt
from q1 import swap_cut_point, binarize, my_method_cut_point
from utils import *

# Images
DOC1_BMP_PATH = "./Imagens/Q2/doc1.bmp"
DOC2_BMP_PATH = "./Imagens/Q2/doc2.bmp"
DOC3_BMP_PATH = "./Imagens/Q2/doc3.bmp"


def normalize(image: np.ndarray) -> tuple[np.ndarray, int, int]:
    max_c = np.max(image.ravel())
    min_c = np.min(image.ravel())

    assert not np.isclose(max_c, min_c)

    return ((image - min_c) / (max_c - min_c)).astype("float64"), max_c, min_c


def desnormalize(image: np.ndarray, max_c: int, min_c: int) -> np.ndarray:
    return np.round(((image * (max_c - min_c)) + min_c)).astype("uint8")


def main():
    images = read_images([DOC1_BMP_PATH, DOC2_BMP_PATH, DOC3_BMP_PATH])
    images_copy = images.copy()
    assert_images_grey_scale(images)

    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    maxs = []
    mins = []

    for i, image in enumerate(images):
        images[i], max_c, min_c = normalize(image)
        maxs.append(max_c)
        mins.append(min_c)

    images = list(map(np.sqrt, images))

    for i, image in enumerate(images):
        images[i] = desnormalize(image, maxs[i], mins[i])

    show_images_hist(images)
    show_images(images)

    show_image(images_copy[0])
    images_copy[0] = binarize(images_copy[0], 50)
    show_image(images_copy[0])

    show_image(images_copy[1])
    images_copy[1] = binarize(images_copy[1], 70)
    show_image(images_copy[1])

    show_image(images_copy[2])
    images_copy[2] = binarize(images_copy[2], my_method_cut_point(images_copy[2]))
    show_image(images_copy[2])


if __name__ == "__main__":
    main()
