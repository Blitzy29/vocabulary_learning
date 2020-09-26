import numpy as np
import pandas as pd
from Levenshtein import distance


def create_vocab_features(vocab):

    vocab['levenshtein_distance_german_english'] = add_levenshtein_distance_german_english(vocab)

    vocab["nb_characters_german"] = vocab["german"].map(len)
    vocab["nb_characters_english"] = vocab["english"].map(len)

    vocab["nb_words_german"] = vocab["german"].map(count_nb_words_german)
    vocab["nb_words_english"] = vocab["english"].map(count_nb_words_english)

    vocab["is_noun"] = vocab.apply(is_noun, axis=1)
    vocab["is_verb"] = vocab.apply(is_verb, axis=1)

    del vocab["german"]
    del vocab["english"]

    return vocab


def remove_article(vocab):

    list_german_article = ['der', 'die', 'das']
    vocab['german'] = vocab['german'].map(
        lambda x: ' '.join(word for word in x.split(' ') if word not in list_german_article)
    )

    list_english_article = ['the', 'to']
    vocab['english'] = vocab['english'].map(
        lambda x: ' '.join(word for word in x.split(' ') if word not in list_english_article)
    )


def add_levenshtein_distance_german_english(vocab):

    vocab = vocab.copy()

    # Lowercase
    vocab['german'] = vocab['german'].str.lower()
    vocab['english'] = vocab['english'].str.lower()

    # Remove article
    remove_article(vocab)

    # Calculate Levenshtein distance
    levenshtein_distance_german_english = vocab.apply(lambda x: distance(x['german'], x['english']), axis=1)

    return levenshtein_distance_german_english


def count_nb_words_german(x):
    list_german_article = ["der", "die", "das"]
    separate_words = x.split(" ")
    if separate_words[0] in list_german_article:
        separate_words = separate_words[1:]
    return len(separate_words)


def count_nb_words_english(x):
    list_english_article = ["the", "to"]
    separate_words = x.split(" ")
    if separate_words[0] in list_english_article:
        separate_words = separate_words[1:]
    return len(separate_words)


def is_noun(x):
    list_german_article = ["der", "die", "das"]
    possible_article = x["german"].split(" ", 1)[0]
    return possible_article in list_german_article


def is_verb(x):
    possible_article = x["english"].split(" ", 1)[0]
    return "to" in possible_article
