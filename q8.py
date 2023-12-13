import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from utils import *

# Images
BINARY_BMP_PATH = "./Imagens/Q8/Binary_Noise.bmp"
CIRCLES_BMP_PATH = "./Imagens/Q8/Circles_Noise.bmp"


# Just complements some binary image
def complement_image(image: np.ndarray) -> np.ndarray:
    return np.abs(255 - image).astype("uint8")


# Applies a dilatation on a 3d image using some structure element
def dilate(image: np.ndarray, structure_element: np.ndarray) -> np.ndarray:
    width, height = structure_element.shape

    half_w = width // 2
    half_h = height // 2

    final_image = np.zeros(image.shape)
    for i in range(len(image)):
        for j in range(len(image[i])):
            image_height = len(image)
            image_width = len(image[0])

            is_active = 0

            for h in range(height):
                for w in range(width):
                    height_bound = i + h - half_h
                    width_bound = j + w - half_w

                    if (
                        height_bound < 0
                        or height_bound >= image_height
                        or width_bound < 0
                        or width_bound >= image_width
                    ):
                        continue

                    if (
                        structure_element[h][w] == image[height_bound][width_bound][0]
                        and structure_element[h][w] == 255
                    ):
                        is_active = 255
                        break

                if is_active:
                    break

            final_image[i][j] = is_active

    return final_image


# Applies a erosion on a 3d matrix using some structure element
def erosion(image: np.ndarray, structure_element: np.ndarray) -> np.ndarray:
    width, height = structure_element.shape

    half_w = width // 2
    half_h = height // 2

    final_image = np.zeros(image.shape)
    for i in range(len(image)):
        for j in range(len(image[i])):
            image_height = len(image)
            image_width = len(image[0])

            is_active = 255

            for h in range(height):
                for w in range(width):
                    height_bound = i + h - half_h
                    width_bound = j + w - half_w

                    if (
                        height_bound < 0
                        or height_bound >= image_height
                        or width_bound < 0
                        or width_bound >= image_width
                    ):
                        continue

                    if (
                        structure_element[h][w] != image[height_bound][width_bound][0]
                        and structure_element[h][w] != 0
                    ):
                        is_active = 0
                        break

                if not is_active:
                    break
            final_image[i][j] = is_active

    return final_image


# Tries to remove noise to the circle image
def remove_circle_noise(circle_image: np.ndarray) -> None:
    # Applies a little circle as a structure element
    circle_structure_element = (
        np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ]
        )
    ) * 255

    # Complement image, the circle aren't the object of interest right now
    circle_image = complement_image(circle_image)

    show_image(circle_image)

    # Applies erosion to remove the little circles outside the circle
    first_circle_erosion = erosion(circle_image, circle_structure_element)

    show_image(first_circle_erosion)

    # Complement the image, now we gonna remove the noise inside the big circle
    second_circle_erosion = erosion(
        complement_image(first_circle_erosion), circle_structure_element
    )

    show_image(second_circle_erosion)

    # Applies another erosion the remove indead what was dilated before
    thirdy_circle_erosion = erosion(second_circle_erosion, circle_structure_element)

    show_image(thirdy_circle_erosion)


# Tries to remove noise to the deer image
def remove_binary_noise(binary_image: np.ndarray) -> None:
    # Applies a diagonal element to flow the direction of the grass
    binary_structure_element = (
        np.array(
            [
                [1, 0],
                [0, 1],
            ]
        )
    ) * 255

    # Opens the image to try to fill the gaps inside the deer
    binary_first_erosion = erosion(binary_image, binary_structure_element)
    binary_opened = dilate(binary_first_erosion, binary_structure_element)

    show_image(binary_opened)

    # Now we gonna work with some circles to  remove the noise from outside the deer
    binary_second_structure_element = (
        np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ]
        )
    ) * 255

    # Closes the image from the gaps on the background
    binary_second_dilated = dilate(binary_opened, binary_second_structure_element)
    binary_closed = erosion(binary_second_dilated, binary_second_structure_element)

    show_image(binary_closed)

    # Applies one more erosion to remove remaing noises
    show_image(erosion(binary_closed, binary_second_structure_element))


# Using median filter
def main2():
    images = read_images([BINARY_BMP_PATH, CIRCLES_BMP_PATH])
    assert_images_grey_scale(images)

    binary_image, circle_image = median_filter(images[0], size=5), median_filter(
        images[1], size=5
    )

    show_image(binary_image)
    show_image(circle_image)


# Using morphology
def main():
    # Read images
    images = read_images([BINARY_BMP_PATH, CIRCLES_BMP_PATH])
    # Assert grey scale
    assert_images_grey_scale(images)

    # Get each individual image to work different
    binary_image, circle_image = images[0], images[1]

    # Tries to remove noise in each image
    remove_binary_noise(binary_image)
    remove_circle_noise(circle_image)


if __name__ == "__main__":
    main()
