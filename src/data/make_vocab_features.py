import numpy as np
import pandas as pd
from Levenshtein import distance


def create_vocab_features(vocab):

    vocab['levenshtein_distance_german_english'] = add_levenshtein_distance_german_english(vocab)

    return vocab


def remove_article(
        vocab,
        list_german_article=['der', 'die', 'das'],
        list_english_article=['the', 'to']
):
    vocab['german'] = vocab['german'].map(
        lambda x: ' '.join(word for word in x.split(' ') if word not in list_german_article)
    )
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
