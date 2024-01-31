from block_type_detector import BlockTypeDetector
from rotator import Rotator
from slicer import Slicer
from utils import (
    get_file_name_from_path,
    process_files_in_folder,
    save_blocks
)


def process_images_in_folder(src):
    file_name = get_file_name_from_path(src)

    rotator = Rotator()
    slicer = Slicer()

    rotated_image = rotator.process_paper_image(src)
    try:
        blocks = slicer.slice(rotated_image)
        type_detector = BlockTypeDetector(blocks)

        test_blocks = type_detector.get_test_blocks()
        numeric_blocks = type_detector.get_numeric_blocks()

        save_blocks(f'images/test/{file_name}', test_blocks)
        save_blocks(f'images/num/{file_name}', numeric_blocks)

    except Exception as ex:
        print(f'Process Failed for {file_name}: {ex}')


def main():
    process_files_in_folder("images/low-res", process_images_in_folder)


if __name__ == "__main__":
    main()
