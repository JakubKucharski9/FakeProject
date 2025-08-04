import os
import shutil
from sklearn.model_selection import train_test_split

# Ścieżki katalogów
data_dir = "C:\\Users\\kuba\\Desktop\\AF1_HF_photos\\AF1Project\\cleaned_unprocessed"
output_dir = "C:\\Users\\kuba\\Desktop\\AF1_HF_photos\\AF1Project\\cleaned"

# Nazwy podkatalogów reprezentujących klasy
include_dirs = ["af1", "af1_fake"]


def train_valid_test_split_and_move(train_size=0.7, valid_size=0.15, test_size=0.15, random_state=None):
    """
    Dzieli zdjęcia na zestawy: treningowy, walidacyjny i testowy oraz przenosi je do odpowiednich katalogów.
    """
    assert train_size + valid_size + test_size == 1, "Rozmiary podziału muszą sumować się do 1"

    for class_dir in include_dirs:
        class_path = os.path.join(data_dir, class_dir)
        images = os.listdir(class_path)

        train, temp = train_test_split(images, test_size=(valid_size + test_size), random_state=random_state)
        valid, test = train_test_split(temp, test_size=(test_size / (valid_size + test_size)),
                                       random_state=random_state)

        for split_name, split_data in zip(["train", "valid", "test"], [train, valid, test]):
            split_dir = os.path.join(output_dir, split_name, class_dir)
            os.makedirs(split_dir, exist_ok=True)

            for img in split_data:
                shutil.copy(os.path.join(class_path, img), os.path.join(split_dir, img))


def main():
    train_valid_test_split_and_move(random_state=42)


if __name__ == "__main__":
    main()
