import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from utils import *

# Images
BINARY_BMP_PATH = "./Imagens/Q8/Binary_Noise.bmp"
CIRCLES_BMP_PATH = "./Imagens/Q8/Circles_Noise.bmp"


def complement_image(image: np.ndarray) -> np.ndarray:
    return np.abs(255 - image).astype("uint8")


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


def remove_circle_noise(circle_image: np.ndarray) -> None:
    circle_structure_element = (
        np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ]
        )
    ) * 255

    circle_image = complement_image(circle_image)

    show_image(circle_image)

    first_circle_erosion = erosion(circle_image, circle_structure_element)

    show_image(first_circle_erosion)

    second_circle_erosion = erosion(
        complement_image(first_circle_erosion), circle_structure_element
    )

    show_image(second_circle_erosion)

    thirdy_circle_erosion = erosion(second_circle_erosion, circle_structure_element)

    show_image(thirdy_circle_erosion)


def remove_binary_noise(binary_image: np.ndarray) -> None:
    binary_structure_element = (
        np.array(
            [
                [1, 0],
                [0, 1],
            ]
        )
    ) * 255

    binary_first_erosion = erosion(binary_image, binary_structure_element)
    binary_opened = dilate(binary_first_erosion, binary_structure_element)

    show_image(binary_opened)

    binary_second_structure_element = (
        np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ]
        )
    ) * 255

    binary_second_dilated = dilate(binary_opened, binary_second_structure_element)
    binary_closed = erosion(binary_second_dilated, binary_second_structure_element)

    show_image(binary_closed)

    show_image(erosion(binary_closed, binary_second_structure_element))


def main():
    images = read_images([BINARY_BMP_PATH, CIRCLES_BMP_PATH])
    assert_images_grey_scale(images)

    binary_image, circle_image = images[0], images[1]

    remove_binary_noise(binary_image)
    remove_circle_noise(circle_image)


if __name__ == "__main__":
    main()
