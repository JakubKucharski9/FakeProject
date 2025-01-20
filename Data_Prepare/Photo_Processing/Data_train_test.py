from nike_pack import *

# Ścieżki katalogów
data_dir = "C:\\Users\\kuba\\Desktop\\test2\\procesed"
output_dir = "C:\\Users\\kuba\\Desktop\\test2\\prepared"

# Nazwy podkatalogów reprezentujących klasy
include_dirs = ["af1", "af1_fake"]

# Parametry podziału danych
test_size = 0.2
random_state = 42

# Ścieżki wyjściowe
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

def copy_files_to_train_test_balanced(data_dir, include_dirs, test_size, random_state):
    """
    Splits files from specified directories into balanced training and testing sets and copies them to respective output directories.

    Parameters:
    - data_dir (str): Path to the main directory containing subdirectories with files to be processed.
    - include_dirs (list of str): List of subdirectory names within data_dir to include in the split.
    - test_size (float): Proportion of files to include in the test split.
    - random_state (int): Seed used by the random number generator for reproducibility.
    """
    # Lista przechowująca pliki dla każdej klasy
    class_files = {}

    for dir_name in include_dirs:
        source_path = os.path.join(data_dir, dir_name)

        if not os.path.exists(source_path):
            print(f"Directory {source_path} does not exist.")
            continue

        files = [os.path.join(source_path, f) for f in os.listdir(source_path) if os.path.isfile(os.path.join(source_path, f))]
        class_files[dir_name] = files

    # Znalezienie minimalnej liczby plików w klasach
    min_files = min(len(files) for files in class_files.values())

    # Przycięcie liczby plików w każdej klasie do min_files
    for dir_name in include_dirs:
        class_files[dir_name] = class_files[dir_name][:min_files]

    for dir_name, files in class_files.items():
        # Podział danych na train i test
        train_files, test_files = train_test_split(files, test_size=test_size, random_state=random_state)

        # Tworzenie podkatalogów w train i test
        train_subdir = os.path.join(train_dir, dir_name)
        test_subdir = os.path.join(test_dir, dir_name)
        os.makedirs(train_subdir, exist_ok=True)
        os.makedirs(test_subdir, exist_ok=True)

        # Kopiowanie plików do train
        for file in train_files:
            shutil.copy(file, train_subdir)

        # Kopiowanie plików do test
        for file in test_files:
            shutil.copy(file, test_subdir)

        print(f"Processed directory: {dir_name}. Train: {len(train_files)}, Test: {len(test_files)}")

copy_files_to_train_test_balanced(data_dir, include_dirs, test_size, random_state)
print("Balanced splitting into train and test completed!")
