import os
import shutil

# Define source and destination directories
source_dir = 'Beans'  # Path to your source folder
destination_dir = '/test_images'  # Path to your destination folder

# Create destination folder if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Get list of all files in the source directory and filter by image extensions
image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']  # List of image extensions
images = [f for f in os.listdir(source_dir) if any(f.lower().endswith(ext) for ext in image_extensions)]
print(images)
# Take first 200 images (or less if there are fewer than 200)
images_to_copy = images[:100]

for image in images_to_copy:
    source_path = os.path.join(source_dir, image)
    destination_path = os.path.join(destination_dir, image)

    # Проверяем, существует ли исходный файл
    if not os.path.exists(source_path):
        print(f"Source file not found: {source_path}")
        continue

    # Копируем файл
    try:
        shutil.copy(source_path, destination_path)
        print(f"Copied {image} to {destination_dir}")
    except Exception as e:
        print(f"Error copying {image}: {e}")

print(f"Copied {len(images_to_copy)} images to {destination_dir}")