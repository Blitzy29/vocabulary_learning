import datetime
import numpy as np
import pandas as pd
import time

from sklearn.dummy import DummyClassifier

import plotly.graph_objects as go


class ModelDummyClassifier:

    def __init__(self):

        self.version = 'dummy_classifier__' + datetime.datetime.today().strftime("%Y%m%d")

        self.global_config = dict()

        self.vardict = self.get_model_vardict()

        self.model_config = {
            'strategy': 'prior'
        }

        self.model = DummyClassifier(**self.model_config)

        self.time_for_training = 0

    def get_model_vardict(self):

        vardict = dict()

        # Target
        vardict["target"] = "result"

        # Numerical
        vardict["numerical"] = [
            #"nb_characters_german",
            #"nb_characters_english",
            "levenshtein_distance_german_english",
            #"previous_score",
            #"previous_question_time",
            "difficulty_category",
        ]

        # Difference in time
        vardict["diff_time"] = [
            #"days_since_last_occurrence_same_language",
            "days_since_first_occur_any_language",
        ]

        # Boolean
        vardict["boolean"] = [
            #"previous_result",
            "is_noun",
        ]

        # Categorical
        vardict["categorical"] = [
            #"language_asked",
            "previous_language_asked",
        ]

        # vardict['all'] = vardict['numerical'] + vardict['diff_time'] + vardict['boolean'] + vardict['categorical']

        return vardict

    def preprocessing_training(self, dataset):

        self.vardict["into_model"] = (
                self.vardict['numerical'] +
                self.vardict['diff_time'] +
                self.vardict['boolean'] +
                self.vardict['categorical']
        )

        return dataset

    def train(self, dataset):

        X_train = dataset[self.vardict["into_model"]]
        y_train = dataset[self.vardict["target"]]

        start = time.time()
        self.model.fit(X_train, y_train)
        end = time.time()

        self.time_for_training = end - start

    def preprocessing_inference(self, dataset):

        return dataset

    def predict(self, dataset, target_present=False):

        X_valid = dataset[self.vardict["into_model"]].copy()

        predictions = X_valid.copy()
        predictions["y_pred"] = self.model.predict(X_valid)
        predictions["y_proba"] = [x[1] for x in self.model.predict_proba(X_valid)]

        if target_present:
            predictions["y_true"] = dataset[self.vardict["target"]].copy()

        return predictions

