import cv2
import matplotlib
import numpy as np
from PIL import Image
from PIL import ImageEnhance
from PIL.Image import Image
from matplotlib import pyplot as plt
import pb1
matplotlib.use('TkAgg')


def enhance_contrast(image):
    """
    Îmbunătățește contrastul imaginii.

    :param image: Imaginea de intrare
    :return: Imaginea cu contrast îmbunătățit
    """
    # Convertirea imaginii în format gri
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicarea transformării de îmbunătățire a contrastului
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray_image)

    enhanced_image = np.power(enhanced_image / 255.0, 1.0) * 255.0

    # Clipping to ensure valid pixel values
    enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)

    return enhanced_image


def edge_detection(image):
    """
    Detectează marginile în imagine folosind operatorul Sobel.

    :param image: Imaginea de intrare
    :return: Imaginea cu margini accentuate
    """
    # Convertirea imaginii în format gri
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicarea operatorului Sobel pentru detectarea marginilor
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    edge_image = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    return edge_image.astype(np.uint8)


def binarization(image, threshold=127):
    """
    Converteste imaginea in alb-negru folosind binarizarea.

    :param image: Imaginea de intrare
    :param threshold: Valoarea prag pentru binarizare (0-255)
    :return: Imaginea binarizata
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image


def enhance_img(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to get a binary image
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)

    # Invert the binary image
    inverted = cv2.bitwise_not(thresh)

    # Apply morphological operations to enhance handwriting
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(inverted, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # Apply bilateral filtering for noise reduction while preserving edges
    enhanced = cv2.bilateralFilter(eroded, 9, 75, 75)

    return enhanced


def enhance_image(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Invert the image
    inverted = cv2.bitwise_not(thresh)

    # Perform morphological operations to remove noise
    kernel = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel)

    # Perform dilation to enhance text
    dilated = cv2.dilate(morph, kernel, iterations=2)

    return dilated


def enhance_handwriting(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding to get a binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Perform morphological operations to remove noise and enhance text
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    sure_bg = cv2.dilate(opening, kernel, iterations=2)

    # Find contours of handwritten text
    contours, _ = cv2.findContours(sure_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to remove small objects
    min_area = 50
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            cv2.drawContours(sure_bg, [contour], 0, (0, 0, 0), -1)

    # Invert the image
    enhanced_image = cv2.bitwise_not(sure_bg)

    return enhanced_image


def remove_spots(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply adaptive thresholding to the image
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Perform morphological operations to remove small black spots
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Invert the result to get the spots in white
    spots = cv2.bitwise_not(opening)

    # Remove the spots from the original image
    cleaned_image = cv2.bitwise_and(img, img, mask=spots)

    return cleaned_image


# image_path = 'data/test2.jpeg'
# input_image = cv2.imread(image_path)
# enhanced_image = enhance_contrast(input_image)
# cv2.imwrite('data/scaled_contrasted_image.jpg', contrasted_image)
#
# input_image_path = 'data/edge_detected_binary_image.jpg'
# enhanced_image = reverse_black_and_white(input_image_path)
# cv2.imwrite('data/2.jpg', enhanced_image)

img = open('data/test2.jpeg', "rb")
pb1.solve(pb1.getDetectedText(img), ["Succes in rezolvarea", "tEMELOR la", "LABORAtoarele de", "Inteligenta Artificiala!"])

print()
print('------------------------------------------------------------------------------')
print()

# img = open('data/contrasted_binary_image.jpg', "rb")
# img = open('data/contrasted_image.jpg', "rb")
img = open('data/edge_detected_binary_image.jpg', "rb")
# img = open('data/edge_detected_image.jpg', "rb")
# img = open('data/enhanced_handwriting.jpg', "rb")
# img = open('data/more_enhanced_handwriting.jpg', "rb")
# img = open('data/test2_contrast.jpeg', "rb")
pb1.solve(pb1.getDetectedText(img), ["Succes in rezolvarea", "tEMELOR la", "LABORAtoarele de", "Inteligenta Artificiala!"])

