import datetime
import dill
import numpy as np
import pandas as pd

import src.data.get_dataset as get_dataset
import src.data.make_dataset as make_dataset

pd.set_option("display.max_columns", None)



def make_predictions_next_session_from_scratch(
        historical_data_path, vocab_path,
        model_path,
        dataset_predictions_path, probas_next_session_path):

    # Create dataset_new_session
    make_dataset.create_dataset_new_session(
        dataset_predictions_path, historical_data_path, vocab_path
    )

    # Take model
    with open(model_path, "rb") as input_file:
        model = dill.load(input_file)

    # Make predictions

    # Get historical data
    with open(dataset_predictions_path, "rb") as input_file:
        dataset_predictions = dill.load(input_file)

    dataset_to_keep = dataset_predictions[
        ["id_vocab", "german_word", "english_word", "language_asked"]
    ]

    dataset_predictions = model.preprocessing_inference(dataset_predictions)
    predictions = model.predict(dataset=dataset_predictions, target_present=False)

    predictions = pd.concat([dataset_to_keep, predictions], axis=1)

    probas_next_session = (
        predictions[["id_vocab", "language_asked", "y_proba"]]
        .pivot(index="id_vocab", columns="language_asked", values="y_proba")
        .reset_index()
    )
    probas_next_session.columns.name = None
    probas_next_session.rename(
        columns={
            "german": "german_proba",
            "english": "english_proba",
        },
        inplace=True,
    )

    # Save dataset
    probas_next_session.to_csv(probas_next_session_path, index=False)
    print(f"Saved at {probas_next_session_path}")


def make_and_save_predictions_next_session(
        model, next_session_features_dataset, probas_next_session_path):

    dataset_to_keep = next_session_features_dataset[
        ["id_vocab", "german_word", "english_word", "language_asked"]
    ]

    dataset_predictions = model.preprocessing_inference(next_session_features_dataset)
    predictions = model.predict(dataset=dataset_predictions, target_present=False)

    predictions = pd.concat([dataset_to_keep, predictions], axis=1)

    probas_next_session = (
        predictions[["id_vocab", "language_asked", "y_proba"]]
        .pivot(index="id_vocab", columns="language_asked", values="y_proba")
        .reset_index()
    )
    probas_next_session.columns.name = None
    probas_next_session.rename(
        columns={
            "german": "german_proba",
            "english": "english_proba",
        },
        inplace=True,
    )

    # Save dataset
    with open(probas_next_session_path, "wb") as file:
        dill.dump(probas_next_session, file)

    print(f"Saved at {probas_next_session_path}")
