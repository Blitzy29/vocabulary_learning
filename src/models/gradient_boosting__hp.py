import dill
import json
import numpy as np
import optuna
import os
import pandas as pd

pd.set_option("display.max_columns", None)

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.gradient_boosting import ModelGradientBoosting
import src.models.performance_metrics as performance_metrics
from src.data.make_dataset import create_folds_for_hyperparameters_tuning


def hyperparameter_objective(trial, list_train_dataset, list_valid_dataset, study_name):
    return optuna_objective(trial, list_train_dataset, list_valid_dataset, study_name)


def optuna_objective(trial, list_train_dataset, list_valid_dataset, study_name):
    """Define hyperparameters choices"""

    model_config_fixed = {
        "objective": "binary",
        "n_estimators": 1000,
        "metrics": "binary_logloss",
        "random_state": 0,
        "verbose": -1,
    }

    model_config_choice = {
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.000001, 1),
        "max_depth": trial.suggest_int("max_depth", 5, 1000, step=5),
        "num_leaves": trial.suggest_int("num_leaves", 2, 1000, step=5),
        "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.1, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 1000, step=1),
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.1, 1.0),
        "early_stopping_round": trial.suggest_int("early_stopping_round", 5, 100, step=5),
    }

    model_config = {**model_config_fixed, **model_config_choice}

    fold_scores = list()

    for fold in range(len(list_train_dataset)):

        model = ModelGradientBoosting()

        model.model_config = model_config

        dataset_processed_train = model.preprocessing_training(list_train_dataset[fold])
        dataset_processed_valid = model.preprocessing_inference(list_valid_dataset[fold])

        model.train(train=dataset_processed_train, valid=dataset_processed_valid)

        y_valid = dataset_processed_valid[model.vardict["target"]].copy()
        predictions = model.predict(dataset=dataset_processed_valid, target_present=False)
        predictions["y_true"] = y_valid.values.tolist()

        binary_classification_results = (
            performance_metrics.get_binary_classification_results(
                predictions,
                model_name=f"{model.version}_valid_{trial.number}_{fold}",
                save_folder=f"data/interim/hyperparameter_tuning/{study_name}"
            )
        )

        regression_results = performance_metrics.get_regression_results(
            predictions,
            model_name=f"{model.version}_valid_{trial.number}_{fold}",
            save_folder=f"data/interim/hyperparameter_tuning/{study_name}"
        )

        fold_scores.append(binary_classification_results["f1_score"])

    return np.array(fold_scores).mean()


def save_best_trial(study, path_best_trial, path_all_trials):

    print("Best hyperparameters: {}".format(study.best_params))

    with open(path_best_trial, "w") as wf:
        wf.write(f"Best score: {study.best_value} at trials {study.best_trial.number}")
        wf.write(json.dumps(study.best_params))

    df_trial = study.trials_dataframe()
    df_trial.to_csv(path_all_trials, index=False)


if __name__ == '__main__':

    path_dataset_train = "data/raw/20210119/dataset_train.pkl"
    path_dataset_valid = "data/raw/20210119/dataset_valid.pkl"

    # get datasets
    with open(path_dataset_train, "rb") as input_file:
        dataset_train = dill.load(input_file)

    with open(path_dataset_valid, "rb") as input_file:
        dataset_valid = dill.load(input_file)

    dataset_hyperoptim = dataset_train.append(dataset_valid)

    # create different training/valid folds
    nb_sessions = max(dataset_hyperoptim["id_session"]) + 1
    nb_folds = 5
    nb_sessions_valid = 2

    list_train_dataset, list_valid_dataset = create_folds_for_hyperparameters_tuning(
        nb_sessions, nb_folds, nb_sessions_valid, dataset_hyperoptim
    )

    # create study
    study = optuna.create_study(direction="maximize", study_name="gb_20210123")

    # save trials
    newpath = f"data/interim/hyperparameter_tuning/{study.study_name}"
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    callback_object = [
        lambda study, trial: save_best_trial(
            study,
            path_best_trial=f"data/interim/hyperparameter_tuning/{study.study_name}/best_param.txt",
            path_all_trials=f"data/interim/hyperparameter_tuning/{study.study_name}/all_trials.csv",
        )
    ]

    # launch hyperparameter tuning
    study.optimize(
        func=lambda x: hyperparameter_objective(x, list_train_dataset, list_valid_dataset, study.study_name),
        n_trials=1000,
        callbacks=callback_object
    )
