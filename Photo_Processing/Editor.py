import os
from rembg import remove
from PIL import Image
import io


def process_images_with_background(input_dir, output_dir):

    """
    Processes all PNG images in the input directory and its subdirectories,
    removing their backgrounds and saving the results with a beige background
    in the output directory.

    Args:
        input_dir (str): Directory containing input images.
        output_dir (str): Directory where processed images will be saved.
    """
    for root, _, files in os.walk(input_dir):

        relative_path = os.path.relpath(root, input_dir)
        output_subdir = os.path.join(output_dir, relative_path)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        for filename in files:
            input_path = os.path.join(root, filename)
            if filename.lower().endswith('.png'):
                with open(input_path, 'rb') as input_file:
                    input_data = input_file.read()
                    output_data = remove(input_data)

                image = Image.open(io.BytesIO(output_data)).convert("RGBA")

                background = Image.new("RGBA", image.size, (245, 245, 220, 255))
                background.paste(image, (0, 0), image)

                output_path = os.path.join(output_subdir, f"{os.path.splitext(filename)[0]}_beige.png")
                background.save(output_path, format="PNG")


input_directory = '../test_data'
output_directory_beige = '../test_output/beige'


process_images_with_background(input_directory, output_directory_beige)

print("Pliki zostały przetworzone.")