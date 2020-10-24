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

    vocab = add_difficulty_category(vocab)

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


def add_difficulty_category(vocab):

    dict_difficulty_category = {
        "Minus10points": -10,
        "Minus9points": -9,
        "Minus8points": -8,
        "Minus7points": -7,
        "Minus6points": -6,
        "Minus5points": -5,
        "Minus4points": -4,
        "Minus3points": -3,
        "Minus2points": -2,
        "Minus1points": -1,
        "0points": 0,
        "1points": 1,
        "2points": 2,
        "3points": 3,
        "4points": 4,
        "5points": 5,
    }

    original_vocab = pd.DataFrame()

    for difficulty_category in dict_difficulty_category.keys():

        i_original_vocab = pd.read_csv(
            f"data/raw/new_vocabulary/{difficulty_category}.csv"
        )
        i_original_vocab["difficulty_category"] = dict_difficulty_category[
            difficulty_category
        ]
        original_vocab = original_vocab.append(i_original_vocab)

    original_vocab = (
        original_vocab.groupby(["German", "English"])
            .agg({"difficulty_category": "max"})
            .reset_index()
    )

    vocab = pd.merge(
        vocab,
        original_vocab[["German", "English", "difficulty_category"]],
        left_on=["german", "english"],
        right_on=["German", "English"],
        how="left",
    )

    del vocab["German"]
    del vocab["English"]

    return vocab
