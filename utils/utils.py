import os

import cv2


# It gets a folder path and call a callback function for each file inside it.
def process_files_in_folder(folder_path, callback):
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    files = os.listdir(folder_path)

    for file in files:
        file_path = os.path.join(folder_path, file)

        if os.path.isfile(file_path):
            callback(file_path)


def get_file_name_from_path(path):
    file_name = os.path.basename(path)
    file_name_without_extension, _ = os.path.splitext(file_name)
    return file_name_without_extension


def get_contours(image, select_top_n=5, gaussian_ksize=(5, 5), canny_min=75, canny_max=200, ):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, gaussian_ksize, 0)

    edged = cv2.Canny(blurred, canny_min, canny_max)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:select_top_n]

    return contours
