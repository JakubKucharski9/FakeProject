from pathlib import Path
import shutil
from PIL import Image
import datetime
import hashlib


paths_and_prefixes = [
    (Path("../Drive/data/af1"), "airforce1_"),
    (Path("../Drive/data/af1_fake"), "fake_airforce1_"),
    (Path("../Drive/data/others"), "other_")
]

backup_root = Path("../Drive/backup")
backup_root.mkdir(parents=True, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

counted_files = 0


def create_backup(source_directory, backup_root, timestamp):
    """
    Creates a backup of the source directory in the backup root directory.

    Args:
        source_directory (Path): Directory to backup.
        backup_root (Path): Directory where backups are stored.
        timestamp (str): Timestamp to use for the backup directory name.

    If the source directory has the same number of files as the latest backup, the backup is not created and
    a message is printed to the console indicating this. Otherwise, the backup is created and a message is printed
    to the console indicating this.
    """
    global changes_detected
    backup_directory = backup_root / f"backup_{timestamp}"

    previous_backups = sorted(backup_root.glob("backup_*"), reverse=True)
    if previous_backups:
        last_backup = previous_backups[0] / source_directory.name
        if last_backup.exists():
            source_file_count = count_files_in_directory(source_directory)
            last_backup_file_count = count_files_in_directory(last_backup)
            if source_file_count == last_backup_file_count:
                print(
                    f"Backup dla {source_directory} nie został utworzony, ponieważ liczba plików jest taka sama jak w "
                    f"poprzednim backupie.")
                return

    changes_detected = True
    backup_directory.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_directory, backup_directory / source_directory.name)
    print(f"Backup utworzony: {backup_directory / source_directory.name}")


def calculate_image_hash(image_path):
    """
    Calculates the MD5 hash of the image at the given path. The hash is calculated
    after resizing the image to 256x256 and converting it to RGB mode.

    Args:
        image_path (Path): Path to the image file.

    Returns:
        str: The MD5 hash of the image as a hexadecimal string.
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize((256, 256))
        hash_md5 = hashlib.md5()
        hash_md5.update(img.tobytes())
        return hash_md5.hexdigest()


def count_files_in_directory(directory):
    """
    Counts the number of files in the specified directory.

    Parameters:
    - directory (Path): The directory in which to count files.

    Returns:
    - int: The total number of files in the directory.
    """
    return sum(1 for file in directory.iterdir() if file.is_file())


def find_available_index(path, prefix):
    """
    Finds the next available index for a filename in the given path.

    The index is determined by looking for the first number after the prefix that is not already used as a filename
    in the given path. If no such index is found, the function will return "1".

    Parameters:
        path (Path): The path in which to search for the next available index.
        prefix (str): The prefix to use when searching for the next available index.

    Returns:
        str: The next available index as a string.
    """
    index = 1
    while any((path / f"{prefix}{index}{file.suffix}").exists() for file in path.iterdir() if file.is_file()):
        index += 1
    return str(index)


def convert_image_to_png(image_path):
    """
    Converts the given image file to PNG format if it is not already a PNG.

    If the image file is in the list of formats specified below, it is converted to PNG format and saved over the original file.
    The list of formats is:

    - .webp
    - .avif
    - .jpg
    - .jpeg
    - .bmp
    - .tiff
    - .gif

    If the image file is not in the list of formats, it is returned unchanged.

    Parameters:
        image_path (Path): The path to the image file to convert.

    Returns:
        Path: The path to the converted image file, or the original image_path if no conversion was done.
    """
    if image_path.suffix.lower() in ['.webp', '.avif', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']:
        png_path = image_path.with_suffix('.png')
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img.save(png_path, 'PNG', optimize=True)
            image_path.unlink()
            return png_path
        except Exception as e:
            print(f"Błąd przy konwersji pliku {image_path}: {e}")
    return image_path


def remove_duplicates(path):
    """
    Removes duplicate files in the given directory.

    This function uses the MD5 hash of the images to determine whether two files are the same. If two files have the same hash,
    the second file is considered a duplicate and is removed.

    Parameters:
        path (Path): The directory to remove duplicates from.

    Returns:
        None
    """
    hashes = {}
    for file_name in path.iterdir():
        if file_name.is_file():
            file_hash = calculate_image_hash(file_name)
            if file_hash in hashes:
                file_name.unlink()
                print(f"Usunięto duplikat: {file_name}")
            else:
                hashes[file_hash] = file_name


def rename_files_in_directory(path, prefix):
    """
    Renames files in the given directory.

    This function goes through all files in the directory and its subdirectories, converts them to PNG format if they are not already in PNG format,
    and renames them to have the given prefix if they do not already have it. The numbering of the files is done so that no two files have the same name.

    Parameters:
        path (Path): The directory to rename files in.
        prefix (str): The prefix to use when renaming files.

    Returns:
        None
    """
    for file_name in path.iterdir():
        if file_name.is_file():
            file_name = convert_image_to_png(file_name)

            if not file_name.name.startswith(prefix):
                new_file_name = file_name.with_name(prefix + find_available_index(path, prefix) + file_name.suffix)
                file_name.rename(new_file_name)


for path, prefix in paths_and_prefixes:
    rename_files_in_directory(path, prefix)

for path, _ in paths_and_prefixes:
    create_backup(path, backup_root, timestamp)

for path, _ in paths_and_prefixes:
    remove_duplicates(path)

for path, _ in paths_and_prefixes:
    files_in_directory = count_files_in_directory(path)
    counted_files += files_in_directory
    print(f"Liczba plików w katalogu {path}: {files_in_directory}")

print(f"Łączna liczba plików: {counted_files}")
