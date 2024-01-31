import os


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
