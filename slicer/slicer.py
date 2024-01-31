import cv2

from utils import get_contours


class Slicer:
    HEADER_ASPECT_RATIO = 108 / 842

    def __init__(self):
        pass

    def _slice_border(self, image, threshold=0.8):
        find_borders = False
        contours = get_contours(image, select_top_n=1, gaussian_ksize=(3, 3))

        for c in contours:
            # Approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, .01 * peri, True)

            if len(approx) == 4:
                contour_area = cv2.contourArea(approx)

                total_area = image.shape[0] * image.shape[1]

                if contour_area / total_area > threshold:
                    find_borders = True

                    x, y, w, h = cv2.boundingRect(approx)

                    cropped_image = image[y:y + h, x:x + w]

                    return cropped_image, find_borders

        return None, find_borders

    def _slice_header(self, image):
        header_height = int(image.shape[0] * self.HEADER_ASPECT_RATIO)

        return image[header_height:]

    def _slice_blocks(self, image, expand_by=20):
        blocks = []

        page_without_header = self._slice_header(image)

        height, width = page_without_header.shape[:2]

        half_width = width // 2

        for i in range(1, -1, -1):
            for j in range(4):
                # Calculate vertical start and end points
                start_y = (height // 4 * j)
                if start_y > expand_by:
                    start_y -= expand_by
                end_y = (height // 4 * (j + 1)) + expand_by

                # Calculate horizontal start and end points
                start_x = (half_width * i)
                if start_x > expand_by:
                    start_x -= expand_by
                end_x = (half_width * (i + 1)) + expand_by

                cropped_piece = page_without_header[start_y:end_y, start_x:end_x]

                blocks.append(cropped_piece)

        return blocks

    def slice(self, image):
        exact_blocks = []

        image, ok = self._slice_border(image)
        if not ok:
            raise Exception("No border found in image")

        blocks = self._slice_blocks(image, expand_by=5)

        for i, block in enumerate(blocks):
            exact_block, ok = self._slice_border(block, threshold=.6)
            if not ok:
                raise Exception("No border found in question block")

            exact_blocks.append(exact_block)

        return exact_blocks
