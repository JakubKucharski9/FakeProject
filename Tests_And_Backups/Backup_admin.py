from __init__ import *

paths_and_prefixes = [
    (Path("../Drive/data/af1"), "airforce1_"),
    (Path("../Drive/data/af1_fake"), "fake_airforce1_"),
    (Path("../Drive/data/others"), "other_")
]

test_output = ["../test_output"]

backup_root = Path("../Drive/backup")
backup_root.mkdir(parents=True, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

counted_files = 0


def count_files_in_directory(directory):
    return sum(1 for _ in directory.rglob('*') if _.is_file())


def find_redundant_backups(backup_root):
    backup_paths = sorted([p for p in backup_root.iterdir() if p.is_dir()])
    redundant_pairs = []
    backup_counts = {}

    for backup in backup_paths:
        backup_file_count = count_files_in_directory(backup)
        if backup_file_count in backup_counts:
            redundant_pairs.append((backup_counts[backup_file_count], backup))
        else:
            backup_counts[backup_file_count] = backup
    return redundant_pairs


def print_redundant_backups(redundant_pairs):
    for original, redundant_backup in redundant_pairs:
        print(f"Zbędny backup: {redundant_backup} (kopię można usunąć, zachowując: {original})")


#Backup creator without looking for duplicates
def create_backup(source_directory, backup_root, timestamp):
    source_directory = Path(source_directory)
    backup_root = Path(backup_root)

    backup_directory = backup_root / f"backup_{timestamp}"
    backup_directory.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_directory, backup_directory / source_directory.name)
    print(f"Backup utworzony: {backup_directory / source_directory.name}")


def backup_on_demand(paths_and_prefixes, backup_root, ts):
    backup_root = Path(backup_root)
    for path, _ in paths_and_prefixes:
        create_backup(path, backup_root, ts)


def main():
    redundant_backups = find_redundant_backups(backup_root)
    if redundant_backups:
        print("Znaleziono redundantne backupy:")
        print_redundant_backups(redundant_backups)
    else:
        print("Nie znaleziono redundantnych backupów.")

    #Backup on demand for data
    #backup_on_demand(paths_and_prefixes, backup_root, timestamp)

    #Backup on demand for test output
    #backup_on_demand(test_output, backup_root, timestamp)


if __name__ == "__main__":
    main()
