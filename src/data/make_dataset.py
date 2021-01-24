import datetime
import dill
import pandas as pd

from sklearn.model_selection import train_test_split

import src.data.get_dataset as get_dataset
from src.data.make_historical_features import create_historical_features

from src.data.get_dataset import get_vocab
from src.data.make_vocab_features import create_vocab_features


def create_dataset(historical_data_path, vocab_path, dataset_path):

    historical_data = get_dataset.get_historical_data(historical_data_path)
    historical_data = create_historical_features(historical_data)

    vocab = get_vocab(vocab_path, list_columns="all")
    vocab = create_vocab_features(vocab)

    dataset = merge_feature_datasets(historical_data, vocab)

    vardict = get_vardict()
    dataset = transform_type(dataset, vardict)

    with open(dataset_path, "wb") as file:
        dill.dump(dataset, file)

    print(f"Saved at {dataset_path}")

    return dataset


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
        "previous_confused_with_another_word",
        "previous_confused_with_an_unknown_word",
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


def split_train_valid_test_dataset(dataset, train_dataset_path, valid_dataset_path, test_dataset_path):
    """ Split dataset into train (70%), valid (20%) and test (10%)
    """

    sessions = list(set(dataset["id_session"].values))

    train_valid_sessions, test_sessions = train_test_split(
        sessions, shuffle=False, test_size=0.10
    )

    train_sessions, valid_sessions = train_test_split(
        train_valid_sessions, shuffle=False, test_size=0.18
    )

    train_dataset = dataset[dataset["id_session"].isin(train_sessions)]
    valid_dataset = dataset[dataset["id_session"].isin(valid_sessions)]
    test_dataset = dataset[dataset["id_session"].isin(test_sessions)]

    print('Training sessions    - {:3d} sessions - {:5d} datapoints.'.format(len(train_sessions), len(train_dataset)))
    print('Validation sessions  - {:3d} sessions - {:5d} datapoints.'.format(len(valid_sessions), len(valid_dataset)))
    print('Test sessions        - {:3d} sessions - {:5d} datapoints.'.format(len(test_sessions), len(test_dataset)))

    with open(train_dataset_path, "wb") as file:
        dill.dump(train_dataset, file)

    with open(valid_dataset_path, "wb") as file:
        dill.dump(valid_dataset, file)

    with open(test_dataset_path, "wb") as file:
        dill.dump(test_dataset, file)

    print('Saved')


def create_historical_data_new_session(vocab_to_predict, max_id_session, max_id_historical_data, language_to):

    fake_historical_data_for_new_session = vocab_to_predict.copy()

    fake_historical_data_for_new_session.rename(
        columns={
            "german": "german_word",
            "english": "english_word",
        },
        inplace=True,
    )

    if language_to == 'german':
        fake_historical_data_for_new_session.rename(
            columns={
                "score_english_german": "score_before",
                "score_german_english": "score_before_other_language",
            },
            inplace=True,
        )
        fake_historical_data_for_new_session["language_asked"] = "german"

    if language_to == 'english':
        fake_historical_data_for_new_session.rename(
            columns={
                "score_english_german": "score_before_other_language",
                "score_german_english": "score_before",
            },
            inplace=True,
        )
        fake_historical_data_for_new_session["language_asked"] = "english"

    del fake_historical_data_for_new_session["try_session_german_english"]
    del fake_historical_data_for_new_session["try_session_english_german"]

    fake_historical_data_for_new_session["guess"] = ""
    fake_historical_data_for_new_session["question_time"] = 0
    fake_historical_data_for_new_session["write_it_again"] = ""
    fake_historical_data_for_new_session["confused_word"] = False
    fake_historical_data_for_new_session["is_it_another_word"] = 0

    fake_historical_data_for_new_session[
        "datetime"
    ] = datetime.datetime.strftime(datetime.datetime.today(), "%Y-%m-%d %H:%M:%S.%f")

    fake_historical_data_for_new_session[
        "day"
    ] = fake_historical_data_for_new_session["datetime"].apply(
        lambda x: datetime.datetime.strftime(
            datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f"), "%Y-%m-%d"
        )
    )

    fake_historical_data_for_new_session[
        "day"
    ] = fake_historical_data_for_new_session["day"].map(
        lambda x: datetime.datetime.strptime(x, "%Y-%m-%d")
    )

    fake_historical_data_for_new_session["id_session"] = max_id_session + 1

    fake_historical_data_for_new_session["id_historical_data"] = list(
        range(
            max_id_historical_data + 1,
            max_id_historical_data
            + 1
            + len(fake_historical_data_for_new_session),
            1,
        )
    )

    return fake_historical_data_for_new_session


def create_dataset_new_session(
    dataset_predictions_path,
    historical_data_path=None, vocab_path=None,
    historical_data=None, vocab_to_predict=None,
):

    if historical_data_path:
        historical_data = get_dataset.get_historical_data(historical_data_path)
    if vocab_path:
        vocab_to_predict = get_dataset.get_vocab(vocab_path, list_columns="all")

    historical_data_new_session_features_german = create_historical_data_new_session_features(
        historical_data, vocab_to_predict, language_to='german'
    )
    historical_data_new_session_features_english = create_historical_data_new_session_features(
        historical_data, vocab_to_predict, language_to='english'
    )

    historical_data_new_session_features = pd.concat([
        historical_data_new_session_features_german, historical_data_new_session_features_english
    ], axis=0)

    # Vocabulary data
    vocab_new_session_features = create_vocab_features(vocab_to_predict)
    vocab_new_session_features = vocab_new_session_features[
        ~vocab_new_session_features["difficulty_category"].isna()
    ]

    # Dataset
    dataset_new_session = merge_feature_datasets(
        historical_data_new_session_features, vocab_new_session_features
    )

    vardict = get_vardict()
    dataset_new_session = transform_type(
        dataset_new_session, vardict
    )

    # Save dataset
    with open(dataset_predictions_path, "wb") as file:
        dill.dump(dataset_new_session, file)
        print(f"Saved at {dataset_predictions_path}")


def create_historical_data_new_session_features(
    historical_data, vocab_to_predict, language_to
):

    max_id_session = max(historical_data["id_session"])
    max_id_historical_data = max(historical_data["id_historical_data"])

    # Historical data
    historical_data_new_session = (
        create_historical_data_new_session(
            vocab_to_predict, max_id_session, max_id_historical_data, language_to
        )
    )

    historical_data_including_new_session = pd.concat(
        [historical_data, historical_data_new_session], axis=0
    )

    historical_data_for_new_session_features = (
        create_historical_features(historical_data_including_new_session)
    )

    historical_data_new_session_features = (
        historical_data_for_new_session_features[
            historical_data_for_new_session_features["id_session"]
            == max_id_session + 1
        ]
    )

    return historical_data_new_session_features


def create_folds_for_hyperparameters_tuning(nb_sessions, nb_folds, nb_sessions_valid, dataset_hyperoptim):

    list_train_dataset = []
    list_valid_dataset = []

    for i_fold in range(nb_folds):
        sessions_train = list(range(nb_sessions))[
                         : (nb_sessions - i_fold * nb_sessions_valid - nb_sessions_valid)
                         ]

        sessions_valid = list(range(nb_sessions))[
                         (nb_sessions - i_fold * nb_sessions_valid - nb_sessions_valid): (
                                 nb_sessions - i_fold * nb_sessions_valid
                         )
                         ]

        dataset_hyperoptim_train = dataset_hyperoptim[
            dataset_hyperoptim["id_session"].isin(sessions_train)
        ]
        dataset_hyperoptim_valid = dataset_hyperoptim[
            dataset_hyperoptim["id_session"].isin(sessions_valid)
        ]

        list_train_dataset.append(dataset_hyperoptim_train)
        list_valid_dataset.append(dataset_hyperoptim_valid)

    for i in range(len(list_train_dataset)):
        print(len(list_train_dataset[i]), len(list_valid_dataset[i]))

    return list_train_dataset, list_valid_dataset
