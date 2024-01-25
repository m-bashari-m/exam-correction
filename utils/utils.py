import os


def get_file_name_from_path(path):
    file_name = os.path.basename(path)
    file_name_without_extension, _ = os.path.splitext(file_name)
    return file_name_without_extension
