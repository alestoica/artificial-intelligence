import random


# def read_corpus(file_name):
#     with open(file_name, 'r', encoding='utf-8') as file:
#         texts = file.read().splitlines()
#
#     return texts


def read_corpus(file_name):
    texts = []

    with open(file_name, 'r', encoding='utf-8') as file:
        texts.append(file.read())

    return texts


def build_markov_chain(texts, state):
    chain = {}

    for text in texts:
        words = text.split()

        for i in range(len(words) - state):
            current_state = ' '.join(words[i:i + state])
            next_word = words[i + state]

            if current_state not in chain:
                chain[current_state] = []

            chain[current_state].append(next_word)

    return chain


def generate_text(chain, length, state):
    initial_state = random.choice(list(chain.keys()))
    generated_words = initial_state.split()

    while len(generated_words) < length:
        current_state = ' '.join(generated_words[-state:])

        if current_state not in chain:
            break

        next_word = random.choice(chain[current_state])
        generated_words.append(next_word)

    return ' '.join(generated_words)


def main():
    # Reading proverbs
    proverbs = read_corpus('data/proverbe.txt')

    # Building the Markov chain
    state = 3
    markov_chain = build_markov_chain(proverbs, state)

    # Generating a proverb
    text_length = 20  # number of words in the proverb
    generated_proverb = generate_text(markov_chain, text_length, state)
    print("Generated proverb:", generated_proverb)


main()
