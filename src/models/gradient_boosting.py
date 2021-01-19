import datetime
import numpy as np
import pandas as pd
import time

import plotly.graph_objects as go

from src.models.performance_metrics import show_results


class ModelGradientBoosting:

    def __init__(self):

        self.version = 'gradient_boosting__' + datetime.datetime.today().strftime("%Y%m%d")

        self.global_config = {
            'show_plot': False,
            'save_plot': True
        }

        self.vardict = self.get_model_vardict()

        self.model_config = {
            'random_state': 0,
            'verbose': 1
        }

        # self.model = LogisticRegression(**self.model_config)

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
                + self.vardict["dummy_boolean"]
                + self.vardict["dummy_categorical"]
        )

        if self.sampling in ['SMOTE']:
            dataset = self.apply_sampling(dataset)

        if self.scaler:
            dataset[self.vardict["preprocessed"]] = self.scaler.fit_transform(dataset[self.vardict["preprocessed"]])

        if self.dimension_reduction:
            dataset = self.apply_dimension_reduction_training(dataset)
        else:
            self.vardict['transformed'] = self.vardict['preprocessed']

        if self.feature_selection in ['Recursive Feature Elimination']:
            self.vardict["into_model"] = self.apply_feature_selection(dataset)
        else:
            self.vardict["into_model"] = self.vardict["transformed"]

        return dataset

    def apply_dimension_reduction_training(self, dataset):

        if not self.scaler:
            raise ValueError("Scaling must be applied before PCA")

        X_train = dataset[self.vardict["preprocessed"]]
        y_train = dataset[self.vardict["target"]]

        X_embedded_pca = self.dimension_reduction.fit_transform(X_train)

        X_train_pca = pd.DataFrame(
            data=X_embedded_pca,
            columns=["component_{:02d}".format(x + 1) for x in range(X_embedded_pca.shape[1])],
        )

        self.vardict['transformed'] = X_train_pca.columns.tolist()
        dataset = pd.concat([X_train_pca, y_train], axis=1)

        if self.global_config['show_plot']:
            pca.show_components(
                dataset, component_a=1, component_b=2,
                model_name=self.version,
                show_plot=self.global_config['show_plot'],
                save_plot=self.global_config['save_plot'],
            )
            pca.plot_explained_variance(pca.explained_variance(X_train_pca), model_name=self.version)
            pca.plot_feature_components(
                feature_components=self.dimension_reduction.components_,
                feature_names=self.vardict["preprocessed"],
                n_components=X_train_pca.shape[1],
                model_name=self.version
            )

        return dataset

    def apply_feature_selection(self, dataset):

        X_train = dataset[self.vardict["transformed"]]
        y_train = dataset[self.vardict["target"]]

        if self.feature_selection == 'Recursive Feature Elimination':

            logreg = LogisticRegression()
            rfe = RFE(estimator=logreg, n_features_to_select=None, verbose=0)
            rfe = rfe.fit(X_train, y_train)

            if self.global_config['show_plot']:
                print("\n ----------")
                print("  Recursive Feature Elimination")
                print(" ----------")
                for i_ranking in range(max(rfe.ranking_)):
                    for feature in [
                        var
                        for var, ranking in zip(self.vardict["preprocessed"], rfe.ranking_)
                        if ranking == i_ranking
                    ]:
                        print("{:2d}. {}".format(i_ranking, feature))
                    if i_ranking == 1:
                        print("\n ---------- \n")

            return [
                var for var, support in zip(self.vardict["preprocessed"], rfe.support_) if support
            ]

    def apply_sampling(self, dataset):

        if self.sampling == 'SMOTE':

            os = SMOTE(
                random_state=0
            )

            n_dataset = len(dataset)
            prop_dataset = np.mean(dataset[self.vardict["target"]])
            X_train = dataset[self.vardict["preprocessed"]]
            y_train = dataset[[self.vardict["target"]]]

            X_train_os, y_train_os = os.fit_sample(X_train, y_train)
            dataset = pd.concat([X_train_os, y_train_os], axis=1)

            # we can Check the numbers of our data
            print("SMOTE - datapoints - {:5d} -> {:5d}".format(
                n_dataset, len(dataset)
            ))
            print("SMOTE - proportion - {:0.3f} -> {:0.3f}".format(
                prop_dataset, np.mean(dataset[self.vardict["target"]])
            ))

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

        if self.scaler:
            dataset[self.vardict["preprocessed"]] = self.scaler.transform(dataset[self.vardict["preprocessed"]])

        #with open(f"data/{self.version}__data_test.pkl", "wb") as file:
        #    dill.dump(dataset, file)

        if self.dimension_reduction:
            dataset = self.apply_dimension_reduction_inference(dataset)

        return dataset

    def apply_dimension_reduction_inference(self, dataset):

        X_inference = dataset[self.vardict["preprocessed"]]

        X_embedded_pca = self.dimension_reduction.transform(X_inference)

        X_inference_pca = pd.DataFrame(
            data=X_embedded_pca,
            columns=self.vardict["transformed"],
        )

        return X_inference_pca

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

    def plot_coefficients(self):

        model_coef = pd.DataFrame()
        model_coef["var"] = self.vardict["into_model"]
        model_coef["coef"] = self.model.coef_.tolist()[0]
        model_coef.sort_values("coef", ascending=True, inplace=True)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=model_coef["coef"].values.tolist(),
                y=model_coef["var"].values.tolist(),
                marker=dict(color="crimson", size=12),
                mode="markers",
                name="Coefficients",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[0] * len(model_coef),
                y=model_coef["var"].values.tolist(),
                mode="lines",
                line=dict(color="black", dash="dash"),
                showlegend=False,
            )
        )

        fig.update_layout(
            title="Coefficients of {}".format(self.version),
            xaxis_title="Coefficients",
            yaxis_title="Variables",
        )

        if self.global_config['show_plot']:
            fig.show()

        if self.global_config['save_plot']:
            fig.write_html(f"data/figures/{self.version}_coefficients.html")

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
