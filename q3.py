import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import *

# Images
ARARAS_BMP_PATH = "./Imagens/Q3/araras.bmp"
F1_BMP_PATH = "./Imagens/Q3/F1.bmp"
GREEN_WATER_BMP_PATH = "./Imagens/Q3/green-water.bmp"
SURF_51_BMP_PATH = "./Imagens/Q3/surf_51.bmp"


def get_colors(image: np.ndarray) -> dict:
    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    colors_map = {}

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if ((b[i][j], g[i][j], r[i][j])) not in colors_map:
                colors_map[(b[i][j], g[i][j], r[i][j])] = 1
            else:
                colors_map[(b[i][j], g[i][j], r[i][j])] += 1

    return colors_map


def reduce_color_range(image: np.ndarray) -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    pallete_size = 8

    pallete = np.linspace(0, 179, endpoint=True, num=pallete_size).astype("uint8")

    for i in range(len(image)):
        for j in range(len(image[i])):
            for p in range(0, len(pallete) - 1, 1):
                if pallete[p] <= image[i][j][0] and image[i][j][0] <= pallete[p + 1]:
                    image[i][j][0] = np.uint8(
                        (int(pallete[p]) + int(pallete[p + 1])) // 2
                    )
                    break

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image


def main():
    images = read_images(
        [ARARAS_BMP_PATH, F1_BMP_PATH, GREEN_WATER_BMP_PATH, SURF_51_BMP_PATH]
    )

    for i, image in enumerate(images):
        colors_size_before = len(get_colors(image))

        images[i] = reduce_color_range(image)

        colors_size_after = len(get_colors(images[i]))

        print(f"Reduction proporsion: {(colors_size_after / colors_size_before):.3f}")

    show_images(images)


if __name__ == "__main__":
    main()
