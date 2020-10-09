import numpy as np
import pandas as pd
import datetime
from Levenshtein import distance


def create_historical_features(historical_data):

    historical_data = add_nb_past_occurrences(historical_data)
    historical_data = add_last_occurrence(historical_data)
    historical_data = add_first_occurrence(historical_data)
    historical_data = add_correct_article(historical_data)
    historical_data = add_levenshtein_distance_guess_answer(historical_data)
    historical_data = add_only_missed_uppercase(historical_data)
    historical_data = add_previous_results(historical_data)
    historical_data = add_write_it_again_features(historical_data)
    historical_data = add_confused_features(historical_data)
    historical_data = add_datetime_features(historical_data)
    historical_data = add_nb_words_same_session(historical_data)

    historical_data = transfer_features_same_language(historical_data)
    historical_data = transfer_features_any_language(historical_data)

    return historical_data


def add_nb_past_occurrences(historical_data):

    historical_data["occurrence"] = 1

    # same language
    past_occurrences = (
        historical_data.groupby(["id_vocab", "language_asked", "id_historical_data"])[
            "occurrence"
        ]
        .sum()
        .groupby(level=[0, 1])
        .cumsum()
        .reset_index()
    )
    past_occurrences.rename(
        columns={"occurrence": "past_occurrences_same_language"}, inplace=True
    )
    past_occurrences["past_occurrences_same_language"] -= 1

    historical_data = pd.merge(
        historical_data,
        past_occurrences[
            ["id_historical_data", "past_occurrences_same_language"]
        ],
        on="id_historical_data",
    )

    historical_data["past_successes_same_language"] = (
        historical_data["past_occurrences_same_language"]
        + historical_data["score_before"]
    ) / 2
    historical_data["past_fails_same_language"] = (
        historical_data["past_occurrences_same_language"]
        - historical_data["score_before"]
    ) / 2

    # any language
    past_occurrences = (
        historical_data.groupby(["id_vocab", "id_historical_data"])["occurrence"]
        .sum()
        .groupby(level=0)
        .cumsum()
        .reset_index()
    )
    past_occurrences.rename(
        columns={"occurrence": "past_occurrences_any_language"}, inplace=True
    )
    past_occurrences["past_occurrences_any_language"] -= 1

    historical_data = pd.merge(
        historical_data,
        past_occurrences[
            ["id_historical_data", "past_occurrences_any_language"]
        ],
        on="id_historical_data",
    )

    historical_data["past_successes_any_language"] = (
        historical_data["past_occurrences_any_language"]
        + (
            historical_data["score_before"]
            + historical_data["score_before_other_language"]
        )
    ) / 2
    historical_data["past_fails_any_language"] = (
        historical_data["past_occurrences_any_language"]
        - (
            historical_data["score_before"]
            + historical_data["score_before_other_language"]
        )
    ) / 2

    del historical_data["occurrence"]

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
        )['day'].idxmin()
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
        )['day'].idxmin()
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

    historical_data['previous_correct_article'] = None

    historical_data.loc[
        (historical_data['language_asked'] == 'german')
        & (historical_data['german_word_article'].isin(list_german_article)),
        'previous_correct_article'
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

    historical_data['previous_levenshtein_distance_guess_answer'] = None
    historical_data.loc[
        historical_data['language_asked'] == 'german',
        "previous_levenshtein_distance_guess_answer"
    ] = historical_data.apply(lambda x: distance(x['german_word_lv'], x['guess_lv']), axis=1)
    historical_data.loc[
        historical_data['language_asked'] == 'english',
        "previous_levenshtein_distance_guess_answer"
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

    historical_data['previous_only_missed_uppercase'] = None

    historical_data.loc[
        historical_data['language_asked'] == 'german', 'previous_only_missed_uppercase'
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
    historical_data['previous_score'] = historical_data['score_before']
    historical_data['previous_score_other_language'] = historical_data['score_before_other_language']
    return historical_data


def add_write_it_again_features(historical_data):

    historical_data['previous_write_it_again_not_null'] = ~historical_data['write_it_again'].isna()

    historical_data.loc[
        historical_data['previous_write_it_again_not_null'], 'previous_write_it_again_german'
    ] = historical_data.loc[
        historical_data['previous_write_it_again_not_null']
    ].apply(
        lambda row: row['write_it_again'].count(row['german_word']), axis=1
    )

    historical_data.loc[
        historical_data['previous_write_it_again_not_null'], 'previous_write_it_again_english'
    ] = historical_data.loc[
        historical_data['previous_write_it_again_not_null']
    ].apply(
        lambda row: row['write_it_again'].count(row['english_word']), axis=1
    )

    return historical_data


def add_datetime_features(historical_data):

    historical_data["week_number"] = historical_data["datetime"].apply(
        lambda x: datetime.datetime.strftime(
            datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f"), "%V"
        )
    )

    historical_data["day_week"] = historical_data["datetime"].apply(
        lambda x: datetime.datetime.strftime(
            datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f"), "%u"
        )
    )

    historical_data["hour"] = historical_data["datetime"].apply(
        lambda x: datetime.datetime.strftime(
            datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f"), "%H"
        )
    )

    return historical_data


def add_nb_words_same_session(historical_data):

    historical_data.sort_values("datetime", inplace=True)
    historical_data["occurrence"] = 1

    nb_words_session = (
        historical_data.groupby(["id_session", "id_historical_data"])["occurrence"]
            .sum()
            .groupby(level=0)
            .cumsum()
            .reset_index()
    )

    nb_words_session.rename(columns={"occurrence": "nb_words_session"}, inplace=True)

    historical_data = pd.merge(
        historical_data,
        nb_words_session[
            ["id_historical_data", "nb_words_session"]
        ],
        on="id_historical_data",
    )

    del historical_data["occurrence"]

    return historical_data


def add_confused_features(historical_data):

    historical_data["previous_confused_with_another_word"] = historical_data[
        "is_it_another_word"
    ] == True

    historical_data.loc[
        historical_data["previous_confused_with_another_word"], "previous_confused_with_an_unknown_word"
    ] = (
            historical_data.loc[historical_data["previous_confused_with_another_word"]][
                "confused_word"
            ]
            == 0
    )

    return historical_data


def transfer_features_same_language(historical_data):

    features_to_transfer = [
        "days_since_first_occur_same_language",
        "previous_result",
        "previous_correct_article",
        "previous_levenshtein_distance_guess_answer",
        "previous_only_missed_uppercase",
        "previous_question_time",
        "previous_write_it_again_not_null",
        "previous_write_it_again_german",
        "previous_write_it_again_english",
        "previous_confused_with_another_word",
        "previous_confused_with_an_unknown_word",
    ]

    prev_occur = historical_data[
        ["id_vocab", "language_asked", "past_occurrences_same_language"]
        + features_to_transfer
    ].copy()
    prev_occur["new_occurr"] = prev_occur["past_occurrences_same_language"] + 1
    del prev_occur["past_occurrences_same_language"]

    for i_feature_to_transfer in features_to_transfer:
        del historical_data[i_feature_to_transfer]

    historical_data = pd.merge(
        historical_data,
        prev_occur,
        left_on=["id_vocab", "language_asked", "past_occurrences_same_language"],
        right_on=["id_vocab", "language_asked", "new_occurr"],
        how="left",
    )

    del historical_data["new_occurr"]

    return historical_data


def transfer_features_any_language(historical_data):

    features_to_transfer = [
        "days_since_first_occur_any_language",
        "previous_language_asked",
    ]

    prev_occur = historical_data[
        ["id_vocab", "past_occurrences_any_language"] + features_to_transfer
    ].copy()
    prev_occur["new_occurr"] = prev_occur["past_occurrences_any_language"] + 1
    del prev_occur["past_occurrences_any_language"]

    for i_feature_to_transfer in features_to_transfer:
        del historical_data[i_feature_to_transfer]

    historical_data = pd.merge(
        historical_data,
        prev_occur,
        left_on=["id_vocab", "past_occurrences_any_language"],
        right_on=["id_vocab", "new_occurr"],
        how="left",
    )

    del historical_data["new_occurr"]

    return historical_data
