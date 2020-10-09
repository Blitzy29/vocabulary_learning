import pandas as pd


def merge_feature_datasets(historical_data, vocab):

    dataset = historical_data.copy()

    dataset = pd.merge(dataset, vocab, on="id_vocab")

    dataset.sort_values("datetime", inplace=True)

    return dataset


def get_vardict():

    vardict = dict()

    # Target
    vardict["target"] = "result"

    # Numerical
    vardict["numerical"] = [
        "nb_characters_german",
        "nb_characters_english",
        "nb_words_german",
        "nb_words_english",
        "levenshtein_distance_german_english",
        "previous_score",
        "previous_score_other_language",
        "previous_levenshtein_distance_guess_answer",
        "previous_question_time",
        "previous_write_it_again_german",
        "previous_write_it_again_english",
        "past_occurrences_same_language",
        "past_successes_same_language",
        "past_fails_same_language",
        "past_occurrences_any_language",
        "past_successes_any_language",
        "past_fails_any_language",
        "week_number",
        "day_week",
        "hour",
        "nb_words_session",
        "difficulty_category",
    ]

    # Difference in time
    vardict["diff_time"] = [
        "days_since_last_occurrence_same_language",
        "days_since_last_occurrence_any_language",
        "days_since_last_success_same_language",
        "days_since_last_success_any_language",
        "days_since_first_occur_same_language",
        "days_since_first_occur_any_language",
    ]

    # Boolean
    vardict["boolean"] = [
        "previous_result",
        "previous_correct_article",
        "previous_only_missed_uppercase",
        "previous_write_it_again_not_null",
        "is_noun",
        "is_verb",
    ]

    # Categorical
    vardict["categorical"] = [
        "language_asked",
        "previous_language_asked",
    ]

    vardict['all'] = vardict['numerical'] + vardict['diff_time'] + vardict['boolean'] + vardict['categorical']

    return vardict


def transform_type(dataset, vardict):

    # Target
    dataset[vardict["target"]] = dataset[vardict["target"]].astype(float)

    # Numerical
    for i_num_var in vardict["numerical"]:
        dataset[i_num_var] = dataset[i_num_var].astype(float)

    # Difference in time
    for i_diff_time_var in vardict["diff_time"]:
        dataset[i_diff_time_var] = dataset[i_diff_time_var].dt.days.astype(float)

    # Boolean
    for i_boolean_var in vardict["boolean"]:

        dataset.loc[~dataset[i_boolean_var].isna(), i_boolean_var] = dataset.loc[
            ~dataset[i_boolean_var].isna(), i_boolean_var
        ].astype(bool)

        dataset.loc[dataset[i_boolean_var].isna(), i_boolean_var] = None

    # Categorical
    for i_categorical_var in vardict["categorical"]:

        dataset.loc[~dataset[i_categorical_var].isna(), i_categorical_var] = dataset.loc[
            ~dataset[i_categorical_var].isna(), i_categorical_var
        ].astype(str)

        dataset.loc[dataset[i_categorical_var].isna(), i_categorical_var] = None

    return dataset
