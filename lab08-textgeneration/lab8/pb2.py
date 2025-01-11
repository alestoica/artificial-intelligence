import random
import markovify
import nltk
from nltk.corpus import wordnet, gutenberg
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
import gensim.downloader as api
from nltk.translate.bleu_score import sentence_bleu
from collections import Counter
from nltk.util import ngrams


# a.
# nltk.download('gutenberg')
def generatePoem():
    gutenberg_corpus = gutenberg.raw()

    text_model = markovify.Text(gutenberg_corpus)
    generated_text = ""

    for i in range(4):
        generated_text += text_model.make_sentence(tries=10) + "\n"

    print("The generated poem:\n", generated_text)

    return generated_text


# b.
# nltk.download('vader_lexicon')
def calculateEmotion(generated_text):
    sia = SentimentIntensityAnalyzer()

    sentiment_scores = sia.polarity_scores(generated_text)

    print("Sentiment Scores:")
    print(f"Positive: {sentiment_scores['pos']}")
    print(f"Negative: {sentiment_scores['neg']}")
    print(f"Neutral: {sentiment_scores['neu']}")
    print(f"Compound: {sentiment_scores['compound']}")

    compound_score = sentiment_scores['compound']

    if compound_score >= 0.05:
        print("The generated text has a positive sentiment.")
    elif compound_score <= -0.05:
        print("The generated text has a negative sentiment.")
    else:
        print("The generated text has a neutral sentiment.")


# c.
# nltk.download('wordnet')
def exchangeWithSynonym(generated_text):
    word2vec_model = api.load('word2vec-google-news-300')

    tokens = word_tokenize(generated_text)

    for token in tokens:
        synonyms = []

        for syn in wordnet.synsets(token):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())

        try:
            embedding_token = word2vec_model[token]
        except KeyError:
            continue

        num_synonyms = 5
        chosen_synonyms = random.sample(synonyms, min(num_synonyms, len(synonyms)))

        min_distance = float('inf')
        chosen_synonym = ""

        for syn in chosen_synonyms:
            try:
                embedding_synonym = word2vec_model[syn]
            except KeyError:
                continue

            distance = abs(embedding_token - embedding_synonym).sum()

            if distance < min_distance:
                min_distance = distance
                chosen_synonym = syn

        if chosen_synonym != "":
            generated_text = generated_text.replace(token, chosen_synonym)

    print("\nThe generated poem with the exchanged synonyms:\n", generated_text)

    return generated_text


# e.
# def calculateBLEU(generated_text, reference_text):
#     gen_tokens = [word_tokenize(sentence) for sentence in generated_text.split('\n') if sentence]
#     ref_tokens = [word_tokenize(sentence) for sentence in reference_text.split('\n') if sentence]
#
#     bleu_score = sentence_bleu(ref_tokens, gen_tokens, weights=(0.25, 0.25, 0.25, 0.25))
#
#     print(f"BLEU Score: {bleu_score}")
def calculateBLEU(generated_text, reference_text):
    # Convert reference and generated texts to lowercase and split into tokens
    reference_tokens = reference_text.lower().split()
    generated_tokens = generated_text.lower().split()

    # Calculate the length of reference and generated texts
    reference_length = len(reference_tokens)
    generated_length = len(generated_tokens)

    # Define the maximum n-gram order for BLEU score calculation
    max_order = 4
    bleu_precisions = []

    # Calculate precision for each n-gram order
    for n in range(1, max_order + 1):
        # Generate n-grams for generated text
        generated_ngrams = list(ngrams(generated_tokens, n))
        generated_ngram_counts = Counter(generated_ngrams)

        # Generate n-grams for reference text
        reference_ngrams = list(ngrams(reference_tokens, n))
        reference_ngram_counts = Counter(reference_ngrams)

        # Calculate the correct counts by taking the minimum of the reference and generated n-gram counts
        correct_counts = sum(min(count, generated_ngram_counts[ng]) for ng, count in reference_ngram_counts.items())

        # Calculate precision for the current n-gram order
        precision = correct_counts / max(1, len(generated_ngrams))

        # Append precision to the list of precisions
        bleu_precisions.append(precision)

    # Calculate the BLEU score using the geometric mean of precisions
    bleu_score = (bleu_precisions[0] * bleu_precisions[1] * bleu_precisions[2] * bleu_precisions[3]) ** (1 / max_order)

    # Print the calculated BLEU score
    print("\nBLEU score: ", bleu_score)


def main():
    generated_poem = generatePoem()
    calculateEmotion(generated_poem)
    generated_poem_with_exchanged_synonyms = exchangeWithSynonym(generated_poem)
    calculateEmotion(generated_poem_with_exchanged_synonyms)
    calculateBLEU(generated_poem_with_exchanged_synonyms, generated_poem)


main()
