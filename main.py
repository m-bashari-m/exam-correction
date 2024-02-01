from preprocessor import preprocessor
from rotator import Rotator
from slicer import Slicer
from utils import (
    get_file_name_from_path
)
from block_type_detector import BlockTypeDetector
from utils import process_files_in_folder, save_blocks

import cv2


def process_images_in_folder(src):
    file_name = get_file_name_from_path(src)

    rotator = Rotator()
    slicer = Slicer()

    rotated_image = rotator.process_paper_image(src)
    cv2.imwrite(f'images/processed/{file_name}_pr.jpg', rotated_image)
    try:
        blocks = slicer.slice(rotated_image)
        type_detector = BlockTypeDetector(blocks)

        # test_blocks = process_image_array(type_detector.get_test_blocks())
        # numeric_blocks = process_image_array(type_detector.get_numeric_blocks())
        test_blocks = type_detector.get_test_blocks()
        numeric_blocks = type_detector.get_numeric_blocks()

        save_blocks(f'images/test/{file_name}', test_blocks)
        save_blocks(f'images/num/{file_name}', numeric_blocks)

        get_numeric_block_results()

    except Exception as ex:
        print(f'Process Failed for {file_name}: {ex}')


def process_image_array(images):
    result = []
    for image in images:
        result.append(preprocessor(image))

    return result


def main():
    process_files_in_folder("images/low-res", process_images_in_folder)
    # rotator = Rotator()
    # slicer = Slicer()
    # image = cv2.imread("images/high-res/1.jpg")
    # rotated_image = rotator.process_paper_image("images/high-res/1.jpg")
    # cv2.imwrite("images/rotated.jpg", rotated_image)
    #
    # processed_image = preprocessor(rotated_image)
    # cv2.imwrite("images/processed.jpg", processed_image)


if __name__ == "__main__":
    main()
