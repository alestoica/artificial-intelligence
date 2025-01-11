import os
from PIL import Image, ImageFilter


def resize_image(input_path, output_path, size):
    # Open the image file
    with Image.open(input_path) as img:
        # Resize the image to a square shape
        img_resized = img.resize((size, size))
        # Apply sepia filter
        img_sepia = sepia_filter(img_resized)
        # Save the processed image
        img_sepia.save(output_path)
        # img_resized.save(output_path)


def sepia_filter(image):
    # Sepia filter constants
    sepia_filter = [
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ]

    # Apply sepia filter to the image
    sepia_img = image.convert("RGB")
    sepia_data = sepia_img.getdata()

    # Apply sepia transformation
    sepia_data = [
        (
            min(int(r * sepia_filter[0][0] + g * sepia_filter[0][1] + b * sepia_filter[0][2]), 255),
            min(int(r * sepia_filter[1][0] + g * sepia_filter[1][1] + b * sepia_filter[1][2]), 255),
            min(int(r * sepia_filter[2][0] + g * sepia_filter[2][1] + b * sepia_filter[2][2]), 255)
        )
        for r, g, b in sepia_data
    ]

    sepia_img.putdata(sepia_data)

    return sepia_img


def process_images(input_dir, output_dir, size=256):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    nr_pic = 1
    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Construct the full file paths
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, "img_sepia" + str(nr_pic) + ".jpg")
            # output_path = os.path.join(output_dir, filename)
            nr_pic += 1

            # Resize and apply sepia filter
            resize_image(input_path, output_path, size)

    print("Image processing completed!")


# Example usage
input_directory = "data"
output_directory = "output_images_64"
image_size = 64

process_images(input_directory, output_directory, image_size)
