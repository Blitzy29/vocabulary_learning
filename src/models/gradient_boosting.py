import datetime
import numpy as np
import pandas as pd
import time

from sklearn.preprocessing import StandardScaler

import lightgbm

import plotly.graph_objects as go

from src.models.performance_metrics import show_results

import src.utils.feature_selection as feature_selection
import src.utils.indexing_variable as indexing_variable
import src.utils.sampling as sampling


class ModelGradientBoosting:

    def __init__(self):

        self.version = 'gradient_boosting__' + datetime.datetime.today().strftime("%Y%m%d")

        self.impute_missing_variables = {
            "previous_levenshtein_distance_guess_answer": -1,
            "previous_question_time": -1,
            "previous_write_it_again_german": -1,
            "previous_write_it_again_english": -1,
            "days_since_last_occurrence_same_language": -1,
            "days_since_last_occurrence_any_language": -1,
            "days_since_last_success_same_language": -1,
            "days_since_last_success_any_language": -1,
            "days_since_first_occur_same_language": -1,
            "days_since_first_occur_any_language": -1,
        }
        self.index_categorical = dict()
        self.index_boolean = dict()

        self.sampling = None  # 'SMOTE'
        self.scaler = None  # StandardScaler()
        self.feature_selection = None  # 'Recursive Feature Elimination'

        self.global_config = {
            'show_plot': False,
            'save_plot': True
        }

        self.vardict = self.get_model_vardict()

        self.model_config = {
            'objective': 'binary',
            'n_estimators': 1000,
            'max_depth': 450,
            'num_leaves': 900,
            'learning_rate': 0.002,
            'feature_fraction': 0.75,
            'bagging_fraction': 0.75,
            'bagging_freq': 400,
            'early_stopping_round': 30,
            'metrics': 'binary_logloss',
            'random_state': 0,
            'verbose': -1
        }

        self.model = None

        self.time_for_training = 0

    def get_model_vardict(self):

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

        # vardict['all'] = vardict['numerical'] + vardict['diff_time'] + vardict['boolean'] + vardict['categorical']

        return vardict

    def preprocessing_training(self, dataset):

        dataset = self.preprocessing_training_numerical(dataset)
        dataset = self.preprocessing_training_diff_time(dataset)
        dataset = self.preprocessing_training_boolean(dataset)
        dataset = self.preprocessing_training_categorical(dataset)

        self.vardict["preprocessed"] = (
                self.vardict["numerical"]
                + self.vardict["diff_time"]
                + self.vardict["boolean"]
                + self.vardict["categorical"]
        )

        if self.sampling in ['SMOTE']:
            dataset = sampling.apply_sampling(dataset)

        if self.scaler:
            dataset[self.vardict["preprocessed"]] = self.scaler.fit_transform(dataset[self.vardict["preprocessed"]])

        self.vardict['transformed'] = self.vardict['preprocessed']

        if self.feature_selection in ['Recursive Feature Elimination']:
            self.vardict["into_model"] = feature_selection.apply_feature_selection(dataset)
        else:
            self.vardict["into_model"] = self.vardict["transformed"]

        return dataset

    def preprocessing_training_numerical(self, dataset):

        list_var_numerical = set(self.impute_missing_variables.keys()).intersection(self.vardict["numerical"])

        for i_var_numerical in list_var_numerical:
            dataset[i_var_numerical].fillna(self.impute_missing_variables[i_var_numerical], inplace=True)

        return dataset

    def preprocessing_training_diff_time(self, dataset):

        list_var_diff_time = set(self.impute_missing_variables.keys()).intersection(self.vardict["diff_time"])

        for i_var_diff_time in list_var_diff_time:
            dataset[i_var_diff_time].fillna(self.impute_missing_variables[i_var_diff_time], inplace=True)

        return dataset

    def preprocessing_training_boolean(self, dataset):

        # LightGBM require label indexing
        # we do not transform as int, as None has an importance: we do not know

        self.index_boolean = indexing_variable.fit_index(
            dataset=dataset,
            list_variables=self.vardict['boolean']
        )

        dataset = indexing_variable.map_to_or_from_index(
            dataset=dataset,
            index=self.index_boolean,
            type_conversion='categorical_to_index'
        )

        return dataset

    def preprocessing_training_categorical(self, dataset):

        # LightGBM require label indexing

        self.index_categorical = indexing_variable.fit_index(
            dataset=dataset,
            list_variables=self.vardict['categorical']
        )

        dataset = indexing_variable.map_to_or_from_index(
            dataset=dataset,
            index=self.index_categorical,
            type_conversion='categorical_to_index'
        )

        return dataset

    def train(self, train, valid=None):

        self.model = lightgbm.LGBMClassifier(**self.model_config)

        eval_set = [(train[self.vardict["into_model"]], train[self.vardict["target"]])]
        eval_names = ['train']
        if valid is not None:
            eval_set += [(valid[self.vardict["into_model"]], valid[self.vardict["target"]])]
            eval_names += ['valid']

        start = time.time()
        self.model.fit(
            X=train[self.vardict["into_model"]],
            y=train[self.vardict["target"]],
            eval_set=eval_set,
            eval_names=eval_names,
            verbose=-1,
        )
        end = time.time()

        self.time_for_training = end - start

    def preprocessing_inference(self, dataset):

        dataset = self.preprocessing_inference_numerical(dataset)
        dataset = self.preprocessing_inference_diff_time(dataset)
        dataset = self.preprocessing_inference_boolean(dataset)
        dataset = self.preprocessing_inference_categorical(dataset)

        if self.scaler:
            dataset[self.vardict["preprocessed"]] = self.scaler.transform(dataset[self.vardict["preprocessed"]])

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

        # LightGBM require label indexing
        dataset = indexing_variable.map_to_or_from_index(
            dataset=dataset,
            index=self.index_boolean,
            type_conversion='categorical_to_index'
        )

        return dataset

    def preprocessing_inference_categorical(self, dataset):

        # LightGBM require label indexing
        dataset = indexing_variable.map_to_or_from_index(
            dataset=dataset,
            index=self.index_categorical,
            type_conversion='categorical_to_index'
        )
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
