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
    # Read images
    images = read_images([BOOK1_BMP_PATH, BOOK2_BMP_PATH])

    # Read structure element
    structure_element = read_images([A_BMP_PATH])[0]

    # Assert grey scale
    assert_images_grey_scale([structure_element])
    assert_images_grey_scale(images)

    # Convert to grey scale to only work with one matrix
    structure_element = cv2.cvtColor(structure_element, cv2.COLOR_BGR2GRAY)

    # Complement structure element
    structure_element = complement_image(structure_element)

    for i, image in enumerate(images):
        # Convert to grey scale to only work with one matrix
        images[i] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Complement image because the letters are the object of interest
        images[i] = complement_image(image)

        # Apply erosion
        erode_image = erosion(images[i], structure_element)

        # Checks if any pixel is a match
        def has_A(image: np.ndarray):
            return np.any(image == 255)

        # Print if it has A or not
        print(f"Image {i} " + ("do has A" if has_A(erode_image) else "don't has A"))


if __name__ == "__main__":
    main()
