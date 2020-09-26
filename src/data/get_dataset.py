import numpy as np
import pandas as pd

import datetime


def get_historical_data(historical_data_path="data/raw/historical_data.csv"):

    historical_data = pd.read_csv(historical_data_path)

    historical_data = define_session(historical_data)

    historical_data["day"] = historical_data["datetime"].apply(
        lambda x: datetime.datetime.strftime(
            datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f"), "%Y-%m-%d"
        )
    )

    historical_data["day"] = historical_data["day"].map(
        lambda x: datetime.datetime.strptime(x, "%Y-%m-%d")
    )

    historical_data["id_historical_data"] = range(len(historical_data))
    historical_data["guess"].fillna("", inplace=True)

    return historical_data


def define_session(historical_data):

    historical_data["datetime_timestamp"] = historical_data["datetime"].apply(
        lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f")
    )

    historical_data["time_since_last_question"] = historical_data[
        "datetime_timestamp"
    ].diff()

    historical_data["id_session"] = (
            historical_data["time_since_last_question"] > datetime.timedelta(hours=1)
    ).cumsum()

    del historical_data["datetime_timestamp"]
    del historical_data["time_since_last_question"]

    return historical_data


def get_vocab(vocab_path="data/raw/german_english.csv", list_columns=None):

    vocab = pd.read_csv(vocab_path)

    if list_columns is None:
        list_columns = ["id_vocab", "german", "english"]
    vocab = vocab[list_columns]

    return vocab


