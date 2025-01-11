from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import matplotlib
import matplotlib.pyplot as plt
import os

matplotlib.use('TkAgg')

subscription_key = "8a1dba3b96cd4b41b16a01b27faecff3"
endpoint = "https://ai-stoica-alexandra.cognitiveservices.azure.com/"
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

image_dir = 'bikes'


def bikeImagesList():
    """
    Returnează o listă cu căile fișierelor imagine care încep cu 'bike' din directorul specificat.

    Args: -

    Returns:
        list: O listă cu căile fișierelor imagine care îndeplinesc condiția specificată.
    """
    imageList = []

    for filename in os.listdir(image_dir):
        image_path = os.path.join(image_dir, filename)

        if filename.startswith('bike'):
            imageList.append(image_path)

    return imageList


def allImagesList():
    """
    Returnează o listă cu căile fișierelor imagine din directorul specificat.

    Args: -

    Returns:
        list: O listă cu căile fișierelor imagine din directorul specificat.
    """
    imageList = []

    for filename in os.listdir(image_dir):
        image_path = os.path.join(image_dir, filename)
        imageList.append(image_path)

    return imageList


bikeImages = bikeImagesList()
allImages = allImagesList()

realLabels = ['bike', 'bike', 'bike', 'bike', 'bike', 'bike', 'bike', 'bike', 'bike', 'bike',
              'no bike', 'no bike', 'no bike', 'no bike', 'no bike', 'no bike', 'no bike', 'no bike',
              'no bike']

computedLabels = []

bike_BB = [[10, 33, 404, 403], [18, 90, 380, 322], [160, 146, 340, 405], [0, 0, 415, 415], [67, 52, 350, 343],
           [10, 33, 404, 403], [57, 205, 294, 415], [55, 5, 390, 348], [8, 19, 375, 376], [145, 124, 367, 404]]


def automaticallyDetect():
    """
    Detectează automat bicicletele în imaginile specificate și afișează rezultatele.

    Args: -

    Returns:
        None
    """
    for i in range(len(bike_BB)):
        real_bike_bb = bike_BB[i]

        predicted_bike_bb = None
        img = open(bikeImages[i], "rb")
        result = computervision_client.analyze_image_in_stream(img, visual_features=[VisualFeatureTypes.objects])
        for ob in result.objects:
            if ob.object_property == "bike" or ob.object_property == "bicycle":
                predicted_bike_bb = [ob.rectangle.x, ob.rectangle.y, ob.rectangle.x + ob.rectangle.w,
                                     ob.rectangle.y + ob.rectangle.h]

        im = plt.imread(bikeImages[i])
        fig = plt.imshow(im)

        fig.axes.add_patch(
            plt.Rectangle(xy=(real_bike_bb[0], real_bike_bb[1]),
                          width=real_bike_bb[2] - real_bike_bb[0], height=real_bike_bb[3] - real_bike_bb[1],
                          fill=False,
                          color="magenta", linewidth=2))

        fig.axes.add_patch(
            plt.Rectangle(xy=(predicted_bike_bb[0], predicted_bike_bb[1]),
                          width=predicted_bike_bb[2] - predicted_bike_bb[0],
                          height=predicted_bike_bb[3] - predicted_bike_bb[1], fill=False, color="green",
                          linewidth=2))

        err = 0
        for v in zip(predicted_bike_bb, real_bike_bb):
            err = err + (v[0] - v[1]) ** 2
        err /= 4
        print("Detection error: ", err)

        plt.show()


def manuallyDetect():
    """
    Detectează manual bicicletele în imaginile specificate și afișează rezultatele.

    Args: -

    Returns:
        None
    """
    for i in range(len(bike_BB)):
        bike_bb = bike_BB[i]
        im = plt.imread(bikeImages[i])
        fig = plt.imshow(im)
        fig.axes.add_patch(
            plt.Rectangle(xy=(bike_bb[0], bike_bb[1]),
                          width=bike_bb[2] - bike_bb[0], height=bike_bb[3] - bike_bb[1],
                          fill=False, color="magenta", linewidth=2))
        plt.show()


def testPerformance1():
    """
    Testează performanța modelului de detecție a bicicletelor pe imaginile din directorul specificat.

    Args: -

    Returns:
        None
    """
    # true_positives = 0
    # false_positives = 0
    # false_negatives = 0
    #
    # for filename in os.listdir(image_dir):
    #     image_path = os.path.join(image_dir, filename)
    #     with open(image_path, "rb") as image_stream:
    #         image_analysis = computervision_client.analyze_image_in_stream(image_stream,
    #                                                                        visual_features=[VisualFeatureTypes.tags])
    #         if any(tag.name == 'bicycle' for tag in image_analysis.tags):
    #             if filename.startswith('bike'):
    #                 true_positives += 1
    #             else:
    #                 false_positives += 1
    #         else:
    #             if filename.startswith('bike'):
    #                 false_negatives += 1
    #
    # precision = true_positives / (true_positives + false_positives)
    # recall = true_positives / (true_positives + false_negatives)
    iouList = testPerformance2()
    count = 0

    for i in range(len(bike_BB)):
        if iouList[i] >= 0.5:
            count += 1

    precision = count / len(bike_BB)

    print('Precision: ', precision)
    # print('Recall: ', recall)


def testPerformance2():
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
    bbox1 (tuple): Coordinates of the first bounding box in format (x1, y1, x2, y2).
    bbox2 (tuple): Coordinates of the second bounding box in format (x1, y1, x2, y2).

    Returns:
    float: Intersection over Union (IoU) value.
    """
    iouList = []

    for i in range(len(bike_BB)):

        x1_1, y1_1, x2_1, y2_1 = bike_BB[i]

        predicted_bike_bb = [0, 0, 0, 0]
        img = open(bikeImages[i], "rb")
        result = computervision_client.analyze_image_in_stream(img, visual_features=[VisualFeatureTypes.objects])
        for ob in result.objects:
            if ob.object_property == "bike" or ob.object_property == "bicycle":
                predicted_bike_bb = [ob.rectangle.x, ob.rectangle.y, ob.rectangle.x + ob.rectangle.w,
                                     ob.rectangle.y + ob.rectangle.h]
        x1_2, y1_2, x2_2, y2_2 = predicted_bike_bb

        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)

        bbox1_area = (x2_1 - x1_1 + 1) * (y2_1 - y1_1 + 1)
        bbox2_area = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1)

        iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
        iouList.append(iou)

    return iouList


# automaticallyDetect()
# manuallyDetect()
# testPerformance1()
print(testPerformance2())
