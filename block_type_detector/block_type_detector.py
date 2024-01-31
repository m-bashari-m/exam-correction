import cv2

from utils import get_contours


class BlockTypeDetector:
    def __init__(self, blocks):
        self._blocks = blocks
        self._test_blocks = []
        self._numeric_blocks = []

        for block in blocks:
            if self._is_test_block(block):
                self._test_blocks.append(block)
            else:
                self._numeric_blocks.append(block)

    def _is_test_block(self, block):
        n_triangles = self._get_number_of_triangles(block)

        if n_triangles > 0:
            return False

        return True

    def _get_number_of_triangles(self, block):
        n_triangles = 0
        contours = get_contours(block, select_top_n=5, gaussian_ksize=(3, 3))

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, .01 * peri, True)

            if len(approx) == 4:
                contour_area = cv2.contourArea(approx)

                total_area = block.shape[0] * block.shape[1]

                if 0.05 < contour_area / total_area < .4:
                    n_triangles += 1

        return n_triangles

    def get_test_blocks(self):
        return self._test_blocks

    def get_numeric_blocks(self):
        return self._numeric_blocks
