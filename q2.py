import numpy as np
import cv2
import matplotlib.pyplot as plt
from q1 import swap_cut_point, binarize, my_method_cut_point
from utils import *

# Images
DOC1_BMP_PATH = "./Imagens/Q2/doc1.bmp"
DOC2_BMP_PATH = "./Imagens/Q2/doc2.bmp"
DOC3_BMP_PATH = "./Imagens/Q2/doc3.bmp"


# Normalize image by it's histogram
def normalize(image: np.ndarray) -> tuple[np.ndarray, int, int]:
    max_c = np.max(image.ravel())
    min_c = np.min(image.ravel())

    assert not np.isclose(max_c, min_c)

    return ((image - min_c) / (max_c - min_c)).astype("float64"), max_c, min_c


# Desnormalize image based on it previous max and min values
def desnormalize(image: np.ndarray, max_c: int, min_c: int) -> np.ndarray:
    return np.round(((image * (max_c - min_c)) + min_c)).astype("uint8")


def main():
    # Load images
    images = read_images([DOC1_BMP_PATH, DOC2_BMP_PATH, DOC3_BMP_PATH])
    
    # Assert if image is grey scale
    assert_images_grey_scale(images)

    # Properly covert to gray scale to only work with one matrix
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get list of max and min values in each histogram
    maxs = []
    mins = []

    for i, image in enumerate(images):
        images[i], max_c, min_c = normalize(image)
        maxs.append(max_c)
        mins.append(min_c)

    # Apply sqrt filter
    images = list(map(np.sqrt, images))

    # Properly covert each image back to original domain
    for i, image in enumerate(images):
        images[i] = desnormalize(image, maxs[i], mins[i])

    # Show histogram of all images
    show_images_hist(images)

    # Gets manual cut point from doc 1 and show images to user
    show_image(images[0])
    images[0] = binarize(images[0], 100)
    show_image(images[0])

    # Gets manual cut point from doc 2 and show images to user
    show_image(images[1])
    images[1] = binarize(images[1], 100)
    show_image(images[1])

    # Swaps cut points with doc 1 and doc 2 and show images to user
    swap_cut_point(images[0], images[1], 100, 100)

    show_image(images[2])
    cut_point = my_method_cut_point(images[2])
    images[2] = binarize(images[2], cut_point)
    show_image(images[2])

    # Show cut point to user
    print(cut_point)


if __name__ == "__main__":
    main()
