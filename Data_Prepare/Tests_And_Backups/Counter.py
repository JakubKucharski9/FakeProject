from __init__ import *


def count_files_in_directory(directory):
    """
    Counts the total number of files in a directory tree.

    Parameters:
    - directory (str): The root directory to start counting from.

    Returns:
    - int: The total number of files in the directory tree.
    """

    file_count = 0
    for root, _, files in os.walk(directory):
        file_count += len(files)
    return file_count

if __name__ == "__main__":
    project_directory = '../Drive'

    total_files = count_files_in_directory(project_directory)
    print(f"Łączna liczba zdjęć w projekcie: {total_files}")
