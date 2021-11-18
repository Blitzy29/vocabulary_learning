import numpy as np
import pandas as pd
from more_itertools import locate, strip


def define_word_language_sessions(i_dataset_vocab, nb_sessions):

    i_id_session = i_dataset_vocab["id_session"].tolist()
    i_result = i_dataset_vocab["result"].tolist()

    i_results_session = np.full(
        shape=nb_sessions,
        fill_value=None,
    ).tolist()

    for r, s in zip(i_result, i_id_session):
        i_results_session[s] = r

    i_results_session = list(
        strip(iterable=i_results_session, pred=lambda x: x in {None})
    )

    return pd.Series(data=[i_results_session], index=["results_session"])


def separate_before_result(i_dataset_vocab):

    i_results_session = i_dataset_vocab["results_session"].tolist()[0]

    return pd.DataFrame.from_dict(
        {"before": [i_results_session[:-1]], "result": i_results_session[-1]}
    )


def multiply_word_language_sessions(i_dataset_vocab):

    i_results_session = i_dataset_vocab["results_session"].tolist()[0]

    all_results_session = [
        (i_results_session[:i], i_results_session[i])
        for i in locate(i_results_session, lambda x: x is not None)
    ]

    session_before = [x[0] for x in all_results_session]
    session_result = [x[1] for x in all_results_session]

    return pd.DataFrame.from_dict({"before": session_before, "result": session_result})


def standardize_sessions(sessions, max_sessions):
    if len(sessions) >= max_sessions:
        return sessions[-max_sessions:]
    else:
        return [None] * (max_sessions - len(sessions)) + sessions


def map_session_to_numeric(x):
    map_to_numeric = {None: 0, 0.0: -1, 1.0: 1}
    return [map_to_numeric[y] for y in x]
