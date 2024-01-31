import cv2
import numpy as np


class Rotator:
    def __init__(self, gaussian_ksize=(5, 5), canny_min=75, canny_max=200, polygon_tolerance=0.2):
        self._gaussian_ksize = gaussian_ksize
        self._canny_min = canny_max
        self._canny_max = canny_max
        self._polygon_tolerance = polygon_tolerance

    def process_paper_image(self, image_path):
        # Read the image
        image = cv2.imread(image_path)
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Noise reduction
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edged = cv2.Canny(blurred, 75, 200)

        # Find contours and sort them by size
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        # Find the contour of the paper
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # If our approximated contour has four points, we assume we have found the paper
            if len(approx) == 4:
                screenCnt = approx
                break

        # Apply the perspective transformation
        pts = screenCnt.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        # The top-left point has the smallest sum, the bottom-right has the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # Compute the difference between the points -- the top-right will have the minumum difference,
        # the bottom-left will have the maximum difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        (tl, tr, br, bl) = rect

        # Compute the width of the new image, which will be the maximum distance between bottom-right and bottom-left
        # x-coordinates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # Compute the height of the new image, which will be the maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # Now that we have the dimensions of the new image, construct the set of destination points
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # Compute the perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        # Rotate if needed to resemble an upright A4 paper
        if maxHeight < maxWidth:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        return warped

# rotation = Rotator()
# processed = rotation.process_paper_image("images/high-res/1.jpg")
# cv2.imwrite("images/processed/1.jpg", processed)
