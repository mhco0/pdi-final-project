import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import *

# Image
MM_BMP_PATH = "./Imagens/Q6/MM.bmp"

def detect_blue(image: np.ndarray) -> bool:
    blue_lower = np.ndarray([100, 100, 100])
    blue_upper = np.ndarray([130, 255, 255])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for i in range(len(image)):
        for j in range(len(image[i])):
            if np.all(image[i][j] >= blue_lower) and np.all(image[i][j] <= blue_upper):
                return True

    return False


def main():
    mm_image = read_images([MM_BMP_PATH])[0]

    if detect_blue(mm_image):
        print(f"It does have blue M&M's")
    else:
        print(f"It doesn't have blue M&M's")

    show_image(mm_image)


if __name__ == "__main__":
    main()
