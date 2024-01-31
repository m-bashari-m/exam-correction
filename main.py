import cv2

from rotator import Rotator
from slicer import Slicer
from utils import get_file_name_from_path, process_files_in_folder


def process_images_in_folder(src):
    file_name = get_file_name_from_path(src)

    rotator = Rotator()
    slicer = Slicer()

    rotated_image = rotator.process_paper_image(src)
    try:
        rotated_image = slicer.slice(rotated_image)
        dest_path = f'images/processed/{file_name}_processed.jpg'
        cv2.imwrite(dest_path, rotated_image)
    except Exception as ex:
        print(f'Process Failed for {file_name}: {ex}')


def main():
    process_files_in_folder("images/low-res", process_images_in_folder)


if __name__ == "__main__":
    main()
