import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load some images
def read_images(paths: list[str]) -> list[np.ndarray]:
    return [cv2.imread(path) for path in paths]

# Save a image
def save_image(image: np.ndarray, path: str) -> None:
    cv2.imwrite(path, image)

# Check if some image is in grey scale
def is_grey_scale(image: np.ndarray) -> None:
    if len(image.shape) < 3:
        return True
    if image.shape[2] == 1:
        return True
    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    if (b == g).all() and (b == r).all():
        return True
    return False

# Assert the images are in grey scale
def assert_images_grey_scale(images: list[np.ndarray]) -> None:
    for image in images:
        assert is_grey_scale(image)


# Applies two_peak algorithm to find cut point on image
def two_peaks(doc: np.ndarray) -> int:
    histogram, bins = np.histogram(doc.ravel(), 256, [0, 256])
    peak_1 = np.argmax(histogram)

    diffs = np.arange(256).astype("float64")

    for k, h_k in enumerate(histogram):
        diffs[k] = ((k - peak_1) ** 2) * h_k

    peak_2 = np.argmax(diffs)

    return (peak_1 + peak_2) // 2


# Show one image on the screen
def show_image(image: np.ndarray) -> None:
    cv2.imshow("QX", image)
    cv2.waitKey(0)

# Show images on the screen
def show_images(images: list[np.ndarray]) -> None:
    for image in images:
        show_image(image)

    cv2.destroyAllWindows()

# Show images histograms
def show_images_hist(images: list[np.ndarray]) -> None:
    for image in images:
        plt.hist(image.ravel(), 256, [0, 256])
        plt.show()
