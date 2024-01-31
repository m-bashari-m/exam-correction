import cv2

from rotator import Rotator
from utils import process_files_in_folder, get_file_name_from_path


def process_images_in_folder(src):
    rotator = Rotator()
    rotated_image = rotator.process_paper_image(src)

    dest_path = f'images/processed/{get_file_name_from_path(src)}_processed.jpg'
    cv2.imwrite(dest_path, rotated_image)


def main():
    process_files_in_folder("images/high-res", process_images_in_folder)


if __name__ == "__main__":
    main()
