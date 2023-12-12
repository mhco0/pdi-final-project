import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import *
from q8 import erosion, complement_image

# Images
BOOK1_BMP_PATH = "./Imagens/Q9/Book_1.bmp"
BOOK2_BMP_PATH = "./Imagens/Q9/Book_2.bmp"

# Structure Element
A_BMP_PATH = "./Imagens/Q9/A.bmp"

def main():
    images = read_images([BOOK1_BMP_PATH, BOOK2_BMP_PATH])

    structure_element = read_images([A_BMP_PATH])[0]

    assert_images_grey_scale([structure_element])
    assert_images_grey_scale(images)

    structure_element = cv2.cvtColor(structure_element, cv2.COLOR_BGR2GRAY)

    structure_element = complement_image(structure_element)

    for i, image in enumerate(images):
        images[i] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images[i] = complement_image(image)

        erode_image = erosion(images[i], structure_element)

        def has_A(image: np.ndarray):
            return np.any(image == 255)

        print(f"Image {i} " + ("do has A" if has_A(erode_image) else "don't has A"))


if __name__ == "__main__":
    main()
