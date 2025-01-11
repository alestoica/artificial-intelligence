from skimage import exposure, io
from skimage.feature import hog
import matplotlib
from PIL import Image, ImageFilter
import os
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def show_image():
    """
    Displays an image.

    Specifications:
    1. Opens and displays an image named 'bike.png' located in the 'data/images' directory.
    2. Uses the default image viewer of the operating system to display the image.

    Dependencies:
    - PIL: for image manipulation.

    Returns:
    None
    """
    image = Image.open("data/images/bike.png")
    image.show()


def resize_images():
    """
    Resizes images and displays them in a grid.

    Specifications:
    1. Reads images from the 'data/images' directory.
    2. Resizes each image to 128x128 pixels.
    3. Displays resized images in a grid layout.
    4. Grid layout is adjusted dynamically based on the number of images.

    Dependencies:
    - PIL: for image manipulation.
    - matplotlib.pyplot: for data visualization.

    Returns:
    None
    """
    files = os.listdir('data/images')
    resized_images = []

    for img in files:
        image = Image.open('data/images/' + img)
        resized_images.append(image.resize((128, 128)))

    num_images = len(resized_images)
    cols = 4
    rows = (num_images + cols - 1) // cols

    plt.figure(figsize=(12, 8))

    for i, img in enumerate(resized_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis('off')

    plt.show()


def gray_levels():
    """
    Converts images to grayscale and displays them in a grid.

    Specifications:
    1. Reads images from the 'data/images' directory.
    2. Converts each image to grayscale.
    3. Displays grayscale images in a grid layout.
    4. Grid layout is adjusted dynamically based on the number of images.

    Dependencies:
    - skimage.io: for image input/output.
    - matplotlib.pyplot: for data visualization.

    Returns:
    None
    """
    files = os.listdir('data/images')
    gray_images = []

    for img in files:
        # image = Image.open('data/images/' + img)
        # gray_image = image.convert("L")
        # gray_images.append(gray_image)
        image = io.imread('data/images/' + img, as_gray=True)
        gray_images.append(image)

    num_images = len(gray_images)
    cols = 4
    rows = (num_images + cols - 1) // cols

    plt.figure(figsize=(12, 8))

    for i, img in enumerate(gray_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')

    plt.show()


def blurr_image():
    """
    Blurs an image and displays both the original and blurred images side by side.

    Specifications:
    1. Opens an image named 'Russell.jpg' located in the 'data/images' directory.
    2. Applies a blur filter to the image.
    3. Displays both the original and blurred images side by side.

    Dependencies:
    - PIL: for image manipulation.
    - matplotlib.pyplot: for data visualization.

    Returns:
    None
    """
    image = Image.open("data/images/Russell.jpg")
    blurred_image = image.filter(ImageFilter.BLUR)

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('imaginea originala')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(blurred_image)
    plt.title('imaginea blurata')
    plt.axis('off')

    plt.show()


def edges1():
    """
    Detects edges in an image and displays both the original and edge-detected images.

    Specifications:
    1. Opens an image named 'Karpaty.jpg' located in the 'data/images' directory.
    2. Detects edges in the image.
    3. Displays both the original and edge-detected images.

    Dependencies:
    - PIL: for image manipulation.
    - matplotlib.pyplot: for data visualization.

    Returns:
    None
    """
    image = Image.open('data/images/Karpaty.jpg')
    edges = image.filter(ImageFilter.FIND_EDGES)
    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('imaginea originala')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(edges)
    plt.title('imaginea cu muchii detectate')
    plt.axis('off')

    plt.show()


def edges2():
    image = io.imread('data/images/Karpaty.jpg', as_gray=True)
    hogDescriptor, hogView = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1),
                                 visualize=True, channel_axis=None)
    hogViewRescaled = exposure.rescale_intensity(hogView, in_range=(0, 10))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 8), sharex=True, sharey=True)
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    ax2.imshow(hogView, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')

    ax3.imshow(hogViewRescaled, cmap=plt.cm.gray)
    ax3.set_title('Histogram of Oriented Gradients (rescaled)')

    plt.show()


show_image()
resize_images()
gray_levels()
blurr_image()
edges1()
edges2()
