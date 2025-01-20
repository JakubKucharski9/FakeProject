from __init__ import *


Drive_data_dir = Path("../Drive/data")
test_output_dir = Path("../test_output")
test_data_dir = Path("../test_data")
beige_data_dir = Path("../Drive/test_output/beige")


def Create_test(input_dir, output_dir):
    """
    Copies data from the input directory to the output directory.

    Parameters:
    - input_dir (str or Path): Directory containing the original data.
    - output_dir (str or Path): Directory where the test data will be copied.

    This function duplicates the contents of the input directory into the output directory.
    If the output directory already exists, the contents will be overwritten.
    """
    shutil.copytree(input_dir, output_dir, dirs_exist_ok=True)
    print("Testowe dane zostały utworzone.")


Create_test(beige_data_dir, test_data_dir)
