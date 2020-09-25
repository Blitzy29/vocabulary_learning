import numpy as np
import pandas as pd
import datetime
from Levenshtein import distance


def create_historical_features(historical_data):

    historical_data = add_nb_previous_occurrences(historical_data)
    historical_data = add_last_occurrence(historical_data)
    historical_data = add_first_occurrence(historical_data)
    historical_data = transfer_features_to_next_datapoint(historical_data)

    return historical_data


def add_nb_previous_occurrences(historical_data):

    historical_data['occurrence'] = 1

    previous_occurrences = historical_data.groupby(
        ['id_vocab', 'language_asked', 'id_historical_data']
    )['occurrence'].sum().groupby(level=[0, 1]).cumsum().reset_index()
    previous_occurrences.rename(columns={
        'occurrence': 'previous_occurrences'
    }, inplace=True)
    previous_occurrences['previous_occurrences'] -= 1

    historical_data = pd.merge(
        historical_data,
        previous_occurrences[['id_historical_data', 'previous_occurrences']],
        on='id_historical_data'
    )

    del historical_data['occurrence']

    historical_data['previous_successes'] = (
        historical_data['previous_occurrences'] + historical_data['score_before']
    ) / 2
    historical_data['previous_fails'] = (
        historical_data['previous_occurrences'] - historical_data['score_before']
    ) / 2

    return historical_data


def add_last_occurrence(historical_data):
    # Calculate the difference between rows - By default, periods = 1
    historical_data['days_since_last_occurrence_same_language'] = historical_data.groupby(
        ['id_vocab', 'language_asked']
    )['day'].diff()

    historical_data['days_since_last_occurrence_any_language'] = historical_data.groupby(
        'id_vocab'
    )['day'].diff()

    historical_data['day_success'] = historical_data['day']
    historical_data.loc[historical_data['result'] != 1, 'day_success'] = None

    # Calculate the difference between rows - By default, periods = 1
    historical_data['days_since_last_success_same_language'] = historical_data.groupby(
        ['id_vocab', 'language_asked']
    )['day_success'].diff()

    historical_data['days_since_last_success_any_language'] = historical_data.groupby(
        'id_vocab'
    )['day_success'].diff()

    del historical_data['day_success']

    return historical_data


def add_first_occurrence(historical_data):

    day_first_occur_same = historical_data.loc[
        historical_data.groupby(
            ['id_vocab', 'language_asked']
        )['day'].idxmax()
    ]
    day_first_occur_same.rename(columns={'day': 'day_first_occur_same_language'}, inplace=True)

    historical_data = pd.merge(
        historical_data,
        day_first_occur_same[['id_vocab', 'language_asked', 'day_first_occur_same_language']],
        on=['id_vocab', 'language_asked'],
        how='left'
    )

    historical_data['days_since_first_occur_same_language'] = (
            historical_data['day'] - historical_data['day_first_occur_same_language']
    )

    del historical_data['day_first_occur_same_language']

    day_first_occur_any = historical_data.loc[
        historical_data.groupby(
            ['id_vocab']
        )['day'].idxmax()
    ]
    day_first_occur_any.rename(columns={'day': 'day_first_occur_any_language'}, inplace=True)

    historical_data = pd.merge(
        historical_data,
        day_first_occur_any[['id_vocab', 'day_first_occur_any_language']],
        on=['id_vocab'],
        how='left'
    )

    historical_data['days_since_first_occur_any_language'] = (
            historical_data['day'] - historical_data['day_first_occur_any_language']
    )

    del historical_data['day_first_occur_any_language']

    return historical_data


def get_german_article(x, list_german_article=['der', 'die', 'das']):
    possible_article = x.split(' ', 1)[0]
    if possible_article in list_german_article:
        return possible_article
    else:
        return None


def add_correct_article(historical_data, list_german_article=['der', 'die', 'das']):
    historical_data['german_word_article'] = None
    historical_data.loc[
        historical_data['language_asked'] == 'german',
        "german_word_article"
    ] = historical_data.loc[
        historical_data['language_asked'] == 'german',
        "german_word"
    ].map(get_german_article)

    historical_data['guess_article'] = None
    historical_data.loc[
        historical_data['language_asked'] == 'german',
        "guess_article"
    ] = historical_data.loc[
        historical_data['language_asked'] == 'german',
        "guess"
    ].map(get_german_article)

    historical_data['correct_article'] = None
    historical_data.loc[
        (historical_data['language_asked'] == 'german')
        & (historical_data['german_word_article'].isin(list_german_article)),
        'correct_article'
    ] = historical_data['german_word_article'] == historical_data['guess_article']

    del historical_data['german_word_article']
    del historical_data['guess_article']

    return historical_data


def add_levenshtein_distance_guess_answer(historical_data):

    list_german_article = ['der', 'die', 'das']
    list_english_article = ['the', 'to']

    # Lowercase
    historical_data['german_word_lv'] = historical_data['german_word'].str.lower()
    historical_data['english_word_lv'] = historical_data['english_word'].str.lower()
    historical_data['guess_lv'] = historical_data['guess'].str.lower()

    historical_data['german_word_lv'] = historical_data['german_word_lv'].map(
        lambda x: ' '.join(word for word in x.split(' ') if word not in list_german_article)
    )
    historical_data['english_word_lv'] = historical_data['english_word_lv'].map(
        lambda x: ' '.join(word for word in x.split(' ') if word not in list_english_article)
    )

    historical_data.loc[
        historical_data['language_asked'] == 'german',
        'guess_lv'
    ] = historical_data['guess_lv'].map(
        lambda x: ' '.join(word for word in x.split(' ') if word not in list_german_article)
    )
    historical_data.loc[
        historical_data['language_asked'] == 'english',
        'guess_lv'
    ] = historical_data['guess_lv'].map(
        lambda x: ' '.join(word for word in x.split(' ') if word not in list_english_article)
    )

    historical_data['levenshtein_distance_guess_answer'] = None
    historical_data.loc[
        historical_data['language_asked'] == 'german',
        "levenshtein_distance_guess_answer"
    ] = historical_data.apply(lambda x: distance(x['german_word_lv'], x['guess_lv']), axis=1)
    historical_data.loc[
        historical_data['language_asked'] == 'english',
        "levenshtein_distance_guess_answer"
    ] = historical_data.apply(lambda x: distance(x['english_word_lv'], x['guess_lv']), axis=1)

    del historical_data['german_word_lv']
    del historical_data['english_word_lv']
    del historical_data['guess_lv']

    return historical_data


def add_only_missed_uppercase(historical_data):
    # Lowercase
    historical_data['german_word_lv'] = historical_data['german_word'].str.lower()

    historical_data['german_word_has_uppercase'] = historical_data['german_word'].map(
        lambda x: any(c.isupper() for c in x)
    )

    historical_data['only_missed_uppercase'] = None

    historical_data.loc[
        historical_data['language_asked'] == 'german', 'only_missed_uppercase'
    ] = (
            historical_data['german_word_has_uppercase']
            & (historical_data['language_asked'] == 'german')
            & (historical_data['german_word_lv'] == historical_data['guess'])
    )

    del historical_data['german_word_lv']
    del historical_data['german_word_has_uppercase']

    return historical_data


def add_previous_results(historical_data):
    previous_results = [
        'language_asked', 'result', 'question_time'
    ]
    for i_previous_col in previous_results:
        historical_data[f'previous_{i_previous_col}'] = historical_data[i_previous_col]
    return historical_data


def add_write_it_again_features(historical_data):

    historical_data['write_it_again_not_null'] = ~historical_data['write_it_again'].isna()

    historical_data.loc[
        historical_data['write_it_again_not_null'], 'write_it_again_german'
    ] = historical_data.loc[
        historical_data['write_it_again_not_null']
    ].apply(
        lambda row: row['write_it_again'].count(row['german_word']), axis=1
    )

    historical_data.loc[
        historical_data['write_it_again_not_null'], 'write_it_again_english'
    ] = historical_data.loc[
        historical_data['write_it_again_not_null']
    ].apply(
        lambda row: row['write_it_again'].count(row['english_word']), axis=1
    )

    return historical_data


def transfer_features_to_next_datapoint(historical_data):
    features_to_transfer = [
        'correct_article', 'levenshtein_distance_guess_answer', 'only_missed_uppercase',
        'previous_language_asked', 'previous_result', 'previous_question_time',
        'write_it_again_not_null', 'write_it_again_german', 'write_it_again_english'
    ]

    historical_data = add_correct_article(historical_data)
    historical_data = add_levenshtein_distance_guess_answer(historical_data)
    historical_data = add_only_missed_uppercase(historical_data)
    historical_data = add_previous_results(historical_data)
    historical_data = add_write_it_again_features(historical_data)

    prev_occur = historical_data[['id_vocab', 'language_asked', 'previous_occurrences'] + features_to_transfer].copy()
    prev_occur['new_occurr'] = prev_occur['previous_occurrences'] + 1
    del prev_occur['previous_occurrences']

    for i_feature_to_transfer in features_to_transfer:
        del historical_data[i_feature_to_transfer]

    historical_data = pd.merge(
        historical_data,
        prev_occur,
        left_on=['id_vocab', 'language_asked', 'previous_occurrences'],
        right_on=['id_vocab', 'language_asked', 'new_occurr'],
        how='left'
    )

    del historical_data['new_occurr']

    return historical_data
