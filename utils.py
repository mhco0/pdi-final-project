import numpy as np
import cv2
import matplotlib.pyplot as plt


def read_images(paths: list[str]) -> list[np.ndarray]:
    return [cv2.imread(path) for path in paths]

def save_image(image: np.ndarray, path: str) -> None:
    cv2.imwrite(path, image)

def is_grey_scale(image: np.ndarray) -> None:
    if len(image.shape) < 3:
        return True
    if image.shape[2] == 1:
        return True
    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    if (b == g).all() and (b == r).all():
        return True
    return False


def assert_images_grey_scale(images: list[np.ndarray]) -> None:
    for image in images:
        assert is_grey_scale(image)


def show_image(image: np.ndarray) -> None:
    cv2.imshow("QX", image)
    cv2.waitKey(0)


def show_images(images: list[np.ndarray]) -> None:
    for image in images:
        show_image(image)

    cv2.destroyAllWindows()


def show_images_hist(images: list[np.ndarray]) -> None:
    for image in images:
        plt.hist(image.ravel(), 256, [0, 256])
        plt.show()