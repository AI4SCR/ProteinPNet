import os
from PIL import Image
from pathlib import Path

# Define your paths
images_folder = Path("CUB_200_2011/images")
bounding_boxes_file = Path("CUB_200_2011/bounding_boxes.txt")
images_txt_file = Path("CUB_200_2011/images.txt")
output_folder = Path("images_cropped")

# Load bounding box data
# Format: {image_id: (x, y, width, height)}
bounding_boxes = {}
with open(bounding_boxes_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        image_id = int(parts[0])
        x, y, width, height = map(float, parts[1:])
        bounding_boxes[image_id] = (x, y, width, height)

# Now process each image
with open(images_txt_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        image_id = int(parts[0])
        image_rel_path = parts[1]

        # Full path to the image
        image_path = images_folder / image_rel_path

        # Load image
        if not image_path.exists():
            print(f"Warning: Image {image_path} not found.")
            continue

        img = Image.open(image_path)

        # Get bounding box
        if image_id not in bounding_boxes:
            print(f"Warning: No bounding box for image ID {image_id}.")
            continue

        x, y, width, height = bounding_boxes[image_id]

        # Crop the image
        cropped_img = img.crop((x, y, x + width, y + height))

        # Prepare output path
        output_path = output_folder / image_rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save cropped image
        cropped_img.save(output_path)

        print(f"Saved cropped image to {output_path}")