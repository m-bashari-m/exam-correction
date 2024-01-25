import cv2 as cv

from utils.utils import get_file_name_from_path


class Cropper:
    # Page height is 842px and header height with it's margin is 108px
    HEADER_ASPECT_RATIO = 108 / 842

    def crop_header(self, page):
        header_height = int(page.shape[0] * self.HEADER_ASPECT_RATIO)

        return page[header_height:]

    def crop(self, src, dest):
        page = cv.imread(src)

        page_without_header = self.crop_header(page)

        height, width = page_without_header.shape[:2]

        half_width = width // 2

        for i in range(2):
            for j in range(4):
                # Calculate vertical start and end points
                start_y = height // 4 * j
                end_y = height // 4 * (j + 1)

                # Calculate horizontal start and end points
                start_x = half_width * i
                end_x = half_width * (i + 1)

                cropped_piece = page_without_header[start_y:end_y, start_x:end_x]

                file_name = get_file_name_from_path(src)
                piece_name = f"{dest}{file_name}_{i}_{j}.png"
                cv.imwrite(piece_name, cropped_piece)
