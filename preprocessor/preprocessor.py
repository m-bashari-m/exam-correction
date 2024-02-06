import cv2


def perform_erosion(binary_image):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5), (2, 2))
    erosion = cv2.erode(binary_image, kernel, iterations=1)
    return erosion


def preprocessor(image, with_erosion=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.ximgproc.niBlackThreshold(gray, 255, cv2.THRESH_BINARY, 41, -0.1,
                                          binarizationMethod=cv2.ximgproc.BINARIZATION_NICK)
    if with_erosion:
        image = perform_erosion(image)

    return image
