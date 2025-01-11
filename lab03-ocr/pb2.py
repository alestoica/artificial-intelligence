from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from array import array
import os
from PIL import Image
import sys
import time
import matplotlib
import cv2
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


subscription_key = 'c0344055cd374f3e979945e0803731c5'
endpoint = 'https://ai-alexandrastoica.cognitiveservices.azure.com/'
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))


def getDetectedTextCoordinates(img):
    """
    Detectează textul dintr-o imagine și returnează coordonatele dreptunghiurilor care încadrează textul detectat.

    Args:
        img (BytesIO): Imaginea din care se va detecta textul.

    Returns:
        List[List[int]]: O listă de liste, fiecare conținând coordonatele (stânga sus, dreapta jos) a unui dreptunghi care încadrează un text detectat.
    """
    read_response = computervision_client.read_in_stream(
        image=img,
        mode="Printed",
        raw=True
    )

    operation_id = read_response.headers['Operation-Location'].split('/')[-1]
    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)

    resultCoordinates = []
    resultLines = []

    print('Textul detectat in imagine: ')
    # resultLines - contine liniile detecatate
    # resultCoordinates - contine coordonatele (stanga sus, dreapta jos)dreptunghiului care incadreaza fiecare linie
    if read_result.status == OperationStatusCodes.succeeded:
        for text_result in read_result.analyze_result.read_results:
            for line in text_result.lines:
                print(line.text)
                resultLines.append(line)
                rect = [line.bounding_box[0], line.bounding_box[1], line.bounding_box[4], line.bounding_box[5]]
                resultCoordinates.append(rect)

    return resultCoordinates


def calculate_overlap(rect1, rect2):
    """
    Calculează coeficientul de suprapunere între două dreptunghiuri.

    Args:
        rect = [x_stanga_sus, y_stanga_sus, x_dreapta_jos, y_dreapta_jos]
        rect1 (List[int]): Coordonatele (stânga sus, dreapta jos) primului dreptunghi (textului detectat).
        rect2 (List[int]): Coordonatele (stânga sus, dreapta jos) celui de-al doilea dreptunghi (textului real).

    Returns:
        float: Coeficientul de suprapunere între cele două dreptunghiuri, ca un procentaj (între 0 și 1).
    """
    left = max(rect1[0], rect2[0])
    top = max(rect1[1], rect2[1])
    right = min(rect1[2], rect2[2])
    bottom = min(rect1[3], rect2[3])

    if left < right and top < bottom:
        intersection_area = (right - left) * (bottom - top)
        rect1_area = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
        overlap_ratio = intersection_area / rect1_area
        return overlap_ratio
    else:
        return 0


points = []


def onclick(event):
    """
    Funcție de handler pentru evenimentul de click pe imagine, pentru a obține coordonatele mouse-ului.
    """
    if event.button == 1:
        points.append(event.xdata)
        points.append(event.ydata)


def solve(image_path):
    points.clear()

    img = open(image_path, "rb")

    rectImg = getDetectedTextCoordinates(img)
    print()
    print('Coordonatele dreptunghiurilor care incadreaza propozitiile detectate in imagine: ')
    print(rectImg)

    print()

    image = cv2.imread(image_path)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Click pe imagine pentru a afla coordonatele mouse-ului.')
    plt.axis('on')

    plt.gcf().canvas.mpl_connect('button_press_event', onclick)

    plt.show()

    rectReal = []
    for i in range(0, len(points), 4):
        rectReal.append([points[i], points[i + 1], points[i + 2], points[i + 3]])

    print('Coordonatele reale ale dreptunghiurilor care incadreaza propozitiile din imagine: ')
    print(rectReal)

    print()

    for i in range(1, len(rectImg) + 1):
        print('Overlap propozitia ', i, ': ', calculate_overlap(rectImg[i - 1], rectReal[i - 1]))


solve("data/test1.png")
print()
print('-----------------------------------------------------------------------------------------------------------')
print()
solve("data/test2.jpeg")

