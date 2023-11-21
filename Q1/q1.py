import numpy as np
import cv2
import matplotlib.pyplot as plt

# Images
DOC1_BMP_PATH = "./Imagens/Q1/doc1.bmp"
DOC2_BMP_PATH = "./Imagens/Q1/doc2.bmp"
DOC3_BMP_PATH = "./Imagens/Q1/doc3.bmp"


def read_images(paths: list[str]) -> list[np.ndarray]:
    return [cv2.imread(path) for path in paths]


def is_grey_scale(image: np.ndarray) -> None:
    if len(image.shape) < 3:
        return True
    if image.shape[2] == 1:
        return True
    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    if (b == g).all() and (b == r).all():
        return True
    return False


def binarize(image: np.ndarray, cut_value: int) -> np.ndarray:
    image[np.where(image <= cut_value)] = 0
    image[np.where(image > cut_value)] = 255

    return image


def show_image(image: np.ndarray) -> None:
    cv2.imshow("Q1", image)
    cv2.waitKey(0)


def show_images(images: list[np.ndarray]) -> None:
    for image in images:
        show_image(image)

    cv2.destroyAllWindows()


def assert_images_grey_scale(images: list[np.ndarray]) -> None:
    for image in images:
        assert is_grey_scale(image)


def show_images_hist(images: list[np.ndarray]) -> None:
    for image in images:
        plt.hist(image.ravel(), 256, [0, 256])
        plt.show()


def swap_cut_point(doc1: np.ndarray, doc2: np.ndarray, cut1: int, cut2: int) -> None:
    show_image(binarize(doc1, cut2))
    show_image(binarize(doc2, cut1))

def main():
    images = read_images([DOC1_BMP_PATH, DOC2_BMP_PATH, DOC3_BMP_PATH])

    assert_images_grey_scale(images)

    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    show_images_hist(images)

    swap_cut_point(image[0], image[1], 80, 70)

    images[0] = binarize(images[0], 80)
    images[1] = binarize(images[1], 70)
    images[2] = binarize(images[2], 150)

    show_images_hist(images)


if __name__ == "__main__":
    main()
