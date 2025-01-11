from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import time


subscription_key = 'c0344055cd374f3e979945e0803731c5'
endpoint = 'https://ai-alexandrastoica.cognitiveservices.azure.com/'
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))


def getDetectedText(img):
    """
    Detectează textul dintr-o imagine și returnează rezultatul.

    Args:
        img (BinaryIO): Imaginea din care se va detecta textul.

    Returns:
        List[str]: O listă de șiruri de caractere reprezentând textul detectat.
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

    result = []
    if read_result.status == OperationStatusCodes.succeeded:
        for text_result in read_result.analyze_result.read_results:
            for line in text_result.lines:
                # print(line.text)
                result.append(line.text)

    return result


def noOfCorrectLines(result, groundTruth):
    """
    Numără câte linii sunt corect detectate.

    Args:
        result (List[str]): Lista rezultatelor detectate.
        groundTruth (List[str]): Lista adevărului de referință.

    Returns:
        int: Numărul de linii corect detectate.
    """
    return sum(i == j for i, j in zip(result, groundTruth))


def getWords(resultText):
    """
    Extrage cuvintele din lista de texte rezultat.

    Args:
        resultText (List[str]): Lista textelor rezultat.

    Returns:
        List[str]: Lista de cuvinte extrase din textul rezultat.
    """
    resultWords = []
    for sentence in resultText:
        [resultWords.append(s) for s in sentence.split(' ')]
    return resultWords


def jaccard_similarity(set1, set2):
    """
    Calculează similitudinea Jaccard între două seturi.
    Calculeaza cat de multe elemente comune au doua seturi, raportat la numarul total de elemente
    distincte in cele doua seturi.

    Args:
        set1 (set): Primul set.
        set2 (set): Al doilea set.

    Returns:
        float: Similitudinea Jaccard între cele două seturi.
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


def word_similarity(str1, str2):
    """
    Calculează Metrica de Similaritate a Cuvintelor între două șiruri de cuvinte.

    Args:
    reference: Textul de referință (șir de cuvinte).
    hypothesis: Textul generat de sistem (șir de cuvinte).

    Returns:
    Metrica de Similaritate a Cuvintelor între cele două șiruri (float).
    """
    common_words = 0

    for word in str1:
        if word in str2:
            common_words += 1

    return float(common_words) / len(str1)


def levenshtein_distance(str1, str2):
    """
    Calculează distanța Levenshtein (edit distance) între două șiruri de caractere.
    Calculeaza numarul minim de operatii necesare pentru a transforma un sir in celalalt.

    Args:
        str1 (str): Primul șir de caractere.
        str2 (str): Al doilea șir de caractere.

    Returns:
        int: Distanța Levenshtein dintre cele două șiruri.
    """
    m = len(str1)
    n = len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


def hamming_distance(str1: str, str2: str) -> int:
    """
    Calculează distanța Hamming între două șiruri de caractere.
    Calculează numărul de poziții în care două șiruri de caractere de aceeași lungime diferă.

    Args:
        str1 (str): Primul șir de caractere.
        str2 (str): Al doilea șir de caractere.

    Returns:
        int: Distanța Hamming dintre cele două șiruri.

    Raises:
        ValueError: Dacă șirurile nu au aceeași lungime.
    """
    if len(str1) != len(str2):
        raise ValueError("Input strings must have the same length")
    return sum(ch1 != ch2 for ch1, ch2 in zip(str1, str2))


def jaro_similarity(str1: str, str2: str) -> float:
    """
    Calculează similitudinea Jaro-Winkler între două șiruri de caractere.

    Măsoară cât de mult se suprapun două șiruri de caractere, luând în considerare numărul
    de caractere comune și distanța lor relativă în cele două șiruri.

    Această metrică acordă o importanță mai mare primelor caractere care se potrivesc
    în cele două șiruri, presupunând că o coincidență la începutul șirului indică o
    similaritate mai mare între șiruri.

    Args:
        str1 (str): Primul șir de caractere.
        str2 (str): Al doilea șir de caractere.

    Returns:
        float: Similitudinea Jaro-Winkler între cele două șiruri.
    """
    if not (str1 and str2):
        return 0

    shorter, longer = str1, str2
    if len(str1) > len(str2):
        longer, shorter = shorter, longer
    match_distance = (max(len(shorter), len(longer)) // 2) - 1
    shorter_matches = [False] * len(shorter)
    longer_matches = [False] * len(longer)
    matches = 0
    transpositions = 0

    for i in range(len(shorter)):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len(longer))
        for j in range(start, end):
            if not longer_matches[j] and shorter[i] == longer[j]:
                shorter_matches[i] = longer_matches[j] = True
                matches += 1
                break

    if matches == 0:
        return 0

    k = 0
    for i in range(len(shorter)):
        if shorter_matches[i]:
            while not longer_matches[k]:
                k += 1
            if shorter[i] != longer[k]:
                transpositions += 1
            k += 1

    return ((matches / len(str1)) +
            (matches / len(str2)) +
            ((matches - transpositions / 2) / matches)) / 3


def longest_common_subsequence(str1: str, str2: str) -> int:
    """
    Calculează lungimea celei mai lungi subsecvențe comune între două șiruri de caractere.

    Args:
        str1 (str): Primul șir de caractere.
        str2 (str): Al doilea șir de caractere.

    Returns:
        int: Lungimea celei mai lungi subsecvențe comune.
    """
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def solve(resultText, groundTruth):
    print('Textul detectat in imagine: ')
    [print(line) for line in resultText]
    print()

    print('Numarul de linii corecte (nivelul propozitie): ', noOfCorrectLines(resultText, groundTruth), '/',
          len(groundTruth))
    print()

    resultWords = getWords(resultText)
    groundTruthWords = getWords(groundTruth)

    print('Coeficientul Jaccard (nivelul cuvant): ', jaccard_similarity(set(resultWords), set(groundTruthWords)))
    print()

    print('Metrica de similaritate a cuvintelor: ', word_similarity(resultWords, groundTruthWords))
    print()

    print('Numarul minim de operatii necesare pentru a transforma sirul detectat in sirul real (nivelul caracter): ',
          levenshtein_distance(str(resultText), str(groundTruth)))
    print()

    try:
        print('Distanta Hamming: ', hamming_distance(str(resultText), str(groundTruth)))
        print()
    except ValueError as ve:
        print('Error: ', ve)
        print()

    print('Distanta Jaro-Winkler: ', jaro_similarity(str(resultText), str(groundTruth)))
    print()

    # print('Longest common subsequence: ', longest_common_subsequence(str(resultText), str(groundTruth)))


img1 = open("data/test1.png", "rb")
resultText1 = getDetectedText(img1)
groundTruth1 = ["Google Cloud", "Platform"]

img2 = open("data/test2.jpeg", "rb")
resultText2 = getDetectedText(img2)
groundTruth2 = ["Succes in rezolvarea", "tEMELOR la", "LABORAtoaree de", "Inteligenta Artificiala!"]

solve(resultText1, groundTruth1)
print()
print('-----------------------------------------------------------------------------------------------------------')
print()
solve(resultText2, groundTruth2)
