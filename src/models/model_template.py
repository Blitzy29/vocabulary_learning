import datetime
import numpy as np
import pandas as pd
import time

import plotly.graph_objects as go

from src.models.performance_metrics import show_results


class ModelSimpleRNN:

    def __init__(self):

        self.version = 'simple_rnn__' + datetime.datetime.today().strftime("%Y%m%d")

        self.global_config = {
            'show_plot': False,
            'save_plot': True
        }

        self.vardict = self.get_model_vardict()

        self.impute_missing_variables = dict()

        self.model_config = {
        }

        # self.model = LogisticRegression(**self.model_config)

        self.time_for_training = 0

    def get_model_vardict(self):

        vardict = dict()

        # Target
        vardict["target"] = "result"

        # Numerical
        vardict["numerical"] = [
            "levenshtein_distance_german_english",
            "difficulty_category",
        ]

        # Difference in time
        vardict["diff_time"] = [
            "days_since_first_occur_any_language",
        ]

        # Boolean
        vardict["boolean"] = [
            "is_noun",
        ]

        # Categorical
        vardict["categorical"] = [
            "previous_language_asked",
        ]

        return vardict

    def preprocessing_training(self, dataset):

        dataset = self.preprocessing_training_numerical(dataset)
        dataset = self.preprocessing_training_diff_time(dataset)
        dataset = self.preprocessing_training_boolean(dataset)
        dataset = self.preprocessing_training_categorical(dataset)

        self.vardict["preprocessed"] = (
                self.vardict["numerical"]
                + self.vardict["diff_time"]
                + self.vardict["dummy_boolean"]
                + self.vardict["dummy_categorical"]
        )

        self.vardict['transformed'] = self.vardict['preprocessed']

        self.vardict["into_model"] = self.vardict["transformed"]

        return dataset

    def preprocessing_training_numerical(self, dataset):

        dataset["previous_levenshtein_distance_guess_answer"].fillna(-1, inplace=True)
        self.impute_missing_variables["previous_levenshtein_distance_guess_answer"] = -1

        dataset["previous_question_time"].fillna(-1, inplace=True)
        self.impute_missing_variables["previous_question_time"] = -1

        dataset["previous_write_it_again_german"].fillna(-1, inplace=True)
        self.impute_missing_variables["previous_write_it_again_german"] = -1

        dataset["previous_write_it_again_english"].fillna(-1, inplace=True)
        self.impute_missing_variables["previous_write_it_again_english"] = -1

        return dataset

    def preprocessing_training_diff_time(self, dataset):

        dataset["days_since_last_occurrence_same_language"].fillna(-1, inplace=True)
        self.impute_missing_variables["days_since_last_occurrence_same_language"] = -1

        dataset["days_since_last_occurrence_any_language"].fillna(-1, inplace=True)
        self.impute_missing_variables["days_since_last_occurrence_any_language"] = -1

        dataset["days_since_last_success_same_language"].fillna(-1, inplace=True)
        self.impute_missing_variables["days_since_last_success_same_language"] = -1

        dataset["days_since_last_success_any_language"].fillna(-1, inplace=True)
        self.impute_missing_variables["days_since_last_success_any_language"] = -1

        dataset["days_since_first_occur_same_language"].fillna(-1, inplace=True)
        self.impute_missing_variables["days_since_first_occur_same_language"] = -1

        dataset["days_since_first_occur_any_language"].fillna(-1, inplace=True)
        self.impute_missing_variables["days_since_first_occur_any_language"] = -1

        return dataset

    def preprocessing_training_boolean(self, dataset):

        # Possibility: label encoding. But IMO does not make sense for LogReg
        # Possibility: have an ordered label encoding
        # https://towardsdatascience.com/logistic-regression-model-tuning-with-scikit-learn-part-1-425142e01af5

        # Transform to dummies

        self.vardict["dummy_boolean"] = []

        for i_var_boolean in self.vardict["boolean"]:

            # possible improvement: pandas.get_dummies(drop_first=False)
            i_dummy_boolean = pd.get_dummies(
                dataset[i_var_boolean],
                prefix=i_var_boolean,
                prefix_sep="__",
                dummy_na=True,
            )

            del dataset[i_var_boolean]

            self.vardict["dummy_boolean"] = (
                    self.vardict["dummy_boolean"] + i_dummy_boolean.columns.tolist()
            )

            dataset = pd.concat([dataset, i_dummy_boolean], axis=1)

        return dataset

    def preprocessing_training_categorical(self, dataset):

        # Transform to dummies

        self.vardict["dummy_categorical"] = []

        for i_var_categorical in self.vardict["categorical"]:

            # possible improvement: pandas.get_dummies(drop_first=False)
            i_dummy_categorical = pd.get_dummies(
                dataset[i_var_categorical],
                prefix=i_var_categorical,
                prefix_sep="__",
                dummy_na=True,
            )

            del dataset[i_var_categorical]

            self.vardict["dummy_categorical"] = (
                    self.vardict["dummy_categorical"] + i_dummy_categorical.columns.tolist()
            )

            dataset = pd.concat([dataset, i_dummy_categorical], axis=1)

        return dataset

    # def hyperparameter_search(self):

    def train(self, dataset):

        X_train = dataset[self.vardict["into_model"]]
        y_train = dataset[self.vardict["target"]]

        start = time.time()
        self.model.fit(X_train, y_train)
        end = time.time()

        self.time_for_training = end - start

    def preprocessing_inference(self, dataset):

        dataset = self.preprocessing_inference_numerical(dataset)
        dataset = self.preprocessing_inference_diff_time(dataset)
        dataset = self.preprocessing_inference_boolean(dataset)
        dataset = self.preprocessing_inference_categorical(dataset)

        return dataset

    def preprocessing_inference_numerical(self, dataset):

        list_var_numerical = set(self.impute_missing_variables.keys()).intersection(self.vardict["numerical"])

        for i_var_numerical in list_var_numerical:
            dataset[i_var_numerical].fillna(self.impute_missing_variables[i_var_numerical], inplace=True)

        return dataset

    def preprocessing_inference_diff_time(self, dataset):

        list_var_diff_time = set(self.impute_missing_variables.keys()).intersection(self.vardict["diff_time"])

        for i_var_diff_time in list_var_diff_time:
            dataset[i_var_diff_time].fillna(self.impute_missing_variables[i_var_diff_time], inplace=True)

        return dataset

    def preprocessing_inference_boolean(self, dataset):

        for i_var_boolean in self.vardict["boolean"]:

            # possible improvement: pandas.get_dummies(drop_first=False)
            i_dummy_boolean = pd.get_dummies(
                dataset[i_var_boolean],
                prefix=i_var_boolean,
                prefix_sep="__",
                dummy_na=True,
            )

            del dataset[i_var_boolean]

            i_dummy_boolean_inference = i_dummy_boolean.columns.tolist()

            i_dummy_boolean_training = [x for x in self.vardict["dummy_boolean"] if x.split("__")[0] == i_var_boolean]

            i_common_dummy_boolean = list(set(i_dummy_boolean_training).intersection(i_dummy_boolean_inference))
            i_missing_dummy_boolean = list(set(i_dummy_boolean_training) - set(i_dummy_boolean_inference))
            i_extra_dummy_boolean = list(set(i_dummy_boolean_inference) - set(i_dummy_boolean_training))

            print(
                "Preprocessing inference - {:40} - {:2d} missing categories".format(
                    i_var_boolean, len(i_extra_dummy_boolean)
                )
            )

            i_dummy_boolean = i_dummy_boolean[i_common_dummy_boolean]
            for i_var_missing_dummy_boolean in i_missing_dummy_boolean:
                i_dummy_boolean[i_var_missing_dummy_boolean] = 0

            dataset = pd.concat([dataset, i_dummy_boolean], axis=1)

        return dataset

    def preprocessing_inference_categorical(self, dataset):

        for i_var_categorical in self.vardict["categorical"]:

            # possible improvement: pandas.get_dummies(drop_first=False)
            i_dummy_categorical = pd.get_dummies(
                dataset[i_var_categorical],
                prefix=i_var_categorical,
                prefix_sep="__",
                dummy_na=True,
            )

            del dataset[i_var_categorical]

            i_dummy_categorical_inference = i_dummy_categorical.columns.tolist()

            i_dummy_categorical_training = [x for x in self.vardict["dummy_categorical"] if x.split("__")[0] == i_var_categorical]

            i_common_dummy_categorical = list(set(i_dummy_categorical_training).intersection(i_dummy_categorical_inference))
            i_missing_dummy_categorical = list(set(i_dummy_categorical_training) - set(i_dummy_categorical_inference))
            i_extra_dummy_categorical = list(set(i_dummy_categorical_inference) - set(i_dummy_categorical_training))

            print(
                "Preprocessing inference - {:40} - {:2d} missing categories".format(
                    i_var_categorical, len(i_extra_dummy_categorical)
                )
            )

            i_dummy_categorical = i_dummy_categorical[i_common_dummy_categorical]
            for i_var_missing_dummy_categorical in i_missing_dummy_categorical:
                i_dummy_categorical[i_var_missing_dummy_categorical] = 0

            dataset = pd.concat([dataset, i_dummy_categorical], axis=1)

        return dataset

    def predict(self, dataset, target_present=False):

        X_valid = dataset[self.vardict["into_model"]].copy()

        predictions = X_valid.copy()
        predictions["y_pred"] = self.model.predict(X_valid)
        predictions["y_proba"] = [x[1] for x in self.model.predict_proba(X_valid)]

        if target_present:
            predictions["y_true"] = dataset[self.vardict["target"]].copy()

        return predictions

    def predict_and_show_results(self, dataset_valid, save_folder="data/processed"):

        y_valid = dataset_valid[self.vardict["target"]].copy()
        dataset_valid = self.preprocessing_inference(dataset_valid)
        predictions = self.predict(dataset=dataset_valid, target_present=False)
        predictions["y_true"] = y_valid.values.tolist()

        show_results(
            predictions,
            model_name=self.version,
            show_plot=self.global_config['show_plot'],
            save_plot=self.global_config['save_plot'],
            save_folder=save_folder,
        )
