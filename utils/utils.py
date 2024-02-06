import os
import cv2
from keras.saving import load_model


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


def get_contours(image, select_top_n=5, gaussian_ksize=(5, 5), canny_min=75, canny_max=200, with_processing=True):
    if with_processing:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, gaussian_ksize, 0)

    edged = cv2.Canny(image, canny_min, canny_max)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:select_top_n]

    return contours


def save_blocks(dest, blocks):
    for i, block in enumerate(blocks):
        cv2.imwrite(f'{dest}_{i}.jpg', block)


def preprocess_for_model(image, resize_to=(40, 60)):
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # image = cv2.resize(image, resize_to)
    image = image.astype('float32') / 255
    image = image.reshape(1, *resize_to)
    return image
