import os

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

subscription_key = "8a1dba3b96cd4b41b16a01b27faecff3"
endpoint = "https://ai-stoica-alexandra.cognitiveservices.azure.com/"
computerVisionClient = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

image_dir = 'bikes'


def classify_images():
    """
    Clasifică imaginile dintr-un director folosind serviciul Computer Vision.

    Args: -

    Returns:
        list: O listă cu etichetele calculate pentru fiecare imagine.
    """
    computedLabels = []

    for filename in os.listdir(image_dir):
        image_path = os.path.join(image_dir, filename)
        with open(image_path, "rb") as image_stream:
            image_analysis = computerVisionClient.analyze_image_in_stream(image_stream,
                                                                          visual_features=[VisualFeatureTypes.tags])
            if any(tag.name == 'bicycle' for tag in image_analysis.tags):
                print(f"{filename}: Contains bicycle")
                computedLabels.append('bike')
            else:
                print(f"{filename}: Does not contain bicycle")
                computedLabels.append('no bike')

    return computedLabels


def evaluate_classification_performance(realLabels, computedLabels, pos, neg):
    """
    Evaluează performanța clasificării pe baza etichetelor reale și calculate.

    Args:
        realLabels (list): Lista cu etichetele reale ale imaginilor.
        computedLabels (list): Lista cu etichetele calculate ale imaginilor.
        pos (str): Eticheta pozitivă (de interes).
        neg (str): Eticheta negativă.

    Returns:
        tuple: Un tuple care conține accuracy, precision și recall.
    """
    acc = sum([1 if realLabels[i] == computedLabels[i] else 0 for i in range(0, len(realLabels))]) / len(realLabels)
    TP = sum([1 if (realLabels[i] == pos and computedLabels[i] == pos) else 0 for i in range(len(realLabels))])
    FP = sum([1 if (realLabels[i] == neg and computedLabels[i] == pos) else 0 for i in range(len(realLabels))])
    # TN = sum([1 if (realLabels[i] == neg and computedLabels[i] == neg) else 0 for i in range(len(realLabels))])
    FN = sum([1 if (realLabels[i] == pos and computedLabels[i] == neg) else 0 for i in range(len(realLabels))])

    precisionPos = TP / (TP + FP)
    recallPos = TP / (TP + FN)
    # precisionNeg = TN / (TN + FN)
    # recallNeg = TN / (TN + FP)

    return acc, precisionPos, recallPos


real = ['bike', 'bike', 'bike', 'bike', 'bike', 'bike', 'bike', 'bike', 'bike', 'bike',
        'no bike', 'no bike', 'no bike', 'no bike', 'no bike', 'no bike', 'no bike', 'no bike',
        'no bike', 'no bike']

# print('Real labels: ', real)
# print()
#
#
computed = classify_images()
#
# print('Computed labels: ', computed)
# print()


accuracy, precision, recall = evaluate_classification_performance(real, computed, 'bike', 'no bike')

print("Performance Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print()
