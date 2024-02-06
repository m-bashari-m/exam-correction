from preprocessor import preprocessor
from rotator import Rotator
from slicer import Slicer
from utils import (
    get_file_name_from_path,
    get_contours
)
from block_type_detector import BlockTypeDetector
from utils import process_files_in_folder, save_blocks
from resutl_extractor import ResultExtractor
from model import Model

import cv2
import numpy as np


def process_images_in_folder(src):
    file_name = get_file_name_from_path(src)

    rotator = Rotator()
    slicer = Slicer()

    rotated_image = rotator.process_paper_image(src)

    try:
        blocks = slicer.slice(rotated_image)
        type_detector = BlockTypeDetector(blocks)

        test_blocks = process_image_array(type_detector.get_test_blocks(), with_erosion=True)
        numeric_blocks = process_image_array(type_detector.get_numeric_blocks(), remove_border=True)

        result_extractor = ResultExtractor(test_blocks, numeric_blocks, model_path="model2.keras")
        save_blocks(f'images/test/{file_name}', test_blocks)
        save_blocks(f'images/num/{file_name}', numeric_blocks)
        print("Test: ", file_name)
        print(result_extractor.get_test_result())
        # print("Numeric", file_name)
        # print(result_extractor.get_numeric_result())

    except Exception as ex:
        print(f'Process Failed for {file_name}: {ex}')


def process_image_array(images, with_erosion=False, remove_border=False, remove_factor=0.05):
    result = []
    for image in images:
        processed = preprocessor(image, with_erosion=with_erosion)
        width, height = processed.shape[0], processed.shape[1]
        if remove_border:
            x_removal = int(remove_factor * width)
            y_removal = int(remove_factor * height)

            result.append(processed[y_removal: -y_removal, x_removal:-x_removal])
        else:
            result.append(processed)

    return result


def main():
    process_files_in_folder("images/a", process_images_in_folder)
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
    # img = cv2.imread("images/num/3.jpg")
    # print(img.shape, "raaf")
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img = cv2.resize(img, (60, 40))
    # img = np.expand_dims(img, 0)
    #
    # print(img.shape, "raaf")
    #
    # model = Model("model.keras")
    # print(model.predict(img))
